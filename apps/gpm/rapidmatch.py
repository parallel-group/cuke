import sys
from itertools import permutations 
import numpy as np
import torch

import run
import codegen.cpu
import transform
from asg import *
from asg2ir import gen_ir
from transform import parallelize


def read_graph(partial_edge):
    np_rowptr = np.fromfile("../citeseer/snap.txt.vertex.bin", dtype=np.int64)
    np_colidx = np.fromfile("../citeseer/snap.txt.edge.bin", dtype=np.int32)

    torch_rowptr = torch.from_numpy(np_rowptr, ).to(torch.int32)
    torch_colidx =torch.from_numpy(np_colidx)
    torch_edge_list = torch.zeros([torch_colidx.shape[0], 2], dtype=torch.int32)

    edge_idx = 0
    for i in range(0, torch_rowptr.shape[0]-1):
        for j in range(torch_rowptr[i].item() , torch_rowptr[i+1].item()):
                if (not partial_edge) or (partial_edge and torch_colidx[j]<i):
                    torch_edge_list[edge_idx][0] = i
                    torch_edge_list[edge_idx][1] = torch_colidx[j]
                    edge_idx = edge_idx+1
    #Cuke
    num_node = torch_rowptr.shape[0]-1
    num_edges = torch_colidx.shape[0]
    if partial_edge:
        num_jobs = edge_idx
    else:
        num_jobs = num_edges

    return torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs


def compute_hash_key(hash_table, node_id):
    key = node_id*2
    if node_id!=0 and hash_table[key][0]!=0:
        while key < hash_table.shape[0]:
            if hash_table[key][0]!=0:
                key+=1
    hash_table[key][0] = node_id
    return key


def build_hasb_table(torch_rowptr):
    num_vertex = torch_rowptr.shape[0]-1

    hash_table = torch.zeros([num_vertex*2, 3], dtype=torch.int32)

    for node_id in range(0, num_vertex):
        key = compute_hash_key(hash_table, node_id)
        hash_table[key][0] = node_id
        hash_table[key][1] = torch_rowptr[node_id]
        hash_table[key][2] = torch_rowptr[node_id+1]

    return hash_table



def read_query_file(filename):
    pmtx = None
    num_node=0
    with open(filename) as p:
        for line in p:
            if line[0]=='v':
                num_node+=1
        p.seek(0, 0)

        pmtx = [[0 for y in range(num_node)] for x in range(num_node)]
        for line in p: 
            if line[0]=='e':
                v0 = int(line[2])
                v1 = int(line[4])
                pmtx[v0][v1] = 1
                pmtx[v1][v0] = 1
        # print("pmtx:")
        # print(pmtx)
        return pmtx

def query_info(pmtx):
    num_edge = 0
    num_node = len(pmtx)
    edge_list = []
    for row_idx in range(0, len(pmtx)):
        for col_idx in range(0, len(pmtx[row_idx])):
            item = pmtx[row_idx][col_idx]
            if item==1 and row_idx<col_idx:
                edge_list.append([row_idx, col_idx])
                num_edge+=1
    
    return num_node, num_edge, edge_list
    

def query_dfs(pmtx):
    edge_order = []

    def _query_dfs(query_adjlist, cur_node, visited):
        if cur_node not in visited:
            visited.add(cur_node)
            for adj_node in query_adjlist[cur_node]:
                if [cur_node, adj_node] not in edge_order and [adj_node, cur_node] not in edge_order:
                    edge_order.append([cur_node, adj_node])
                _query_dfs(query_adjlist, adj_node, visited)
    
    query_adjlist = []

    for row_idx in range(0, len(pmtx)):
        this_list = []
        for col_idx in range(0, len(pmtx[row_idx])):
            if pmtx[row_idx][col_idx]==1:
                this_list.append(col_idx)
        
        query_adjlist.append(this_list)

    _query_dfs(query_adjlist, 0, set())
    return edge_order



def node_degree(pmtx, idx):
    res = 0
    row = pmtx[idx]
    for i in row:
        if i==1:
            res+=1
    return res


def breaksymmetry(a, val):
    c = a.apply(lambda x: smaller(x, val))
    return a.apply(lambda x: x, cond=c)

def intersect(a, b):
    src = inspect.cleandoc("""
    RES_SIZE = SetIntersection(FIRST_TENSOR, FIRST_SIZE, SECOND_TENSOR, SECOND_SIZE, RES_TENSOR);
    """)
    output_size = Var(dtype='int')
    output_size.attr['is_arg'] = False

    output_tensor = Tensor((4096, ), dtype='int')
    output_tensor.attr['is_arg'] = False

    output_size = inline(src, ('RES_SIZE', output_size), \
                                ('FIRST_TENSOR', a[0]), ('FIRST_SIZE', a._size()[0]), \
                                ('SECOND_TENSOR', b[0]), ('SECOND_SIZE',  b._size()[0]),  \
                                ('RES_TENSOR', output_tensor[0]))
    ret_val = output_tensor[0:output_size]
    ret_val.ref_size[0].attr['dynamic_size'] = True
    ret_val.attr['is_set'] = True
    return ret_val

def difference(a, b):
    src = inspect.cleandoc("""
    RES_SIZE = SetDifference(FIRST_TENSOR, FIRST_SIZE, SECOND_TENSOR, SECOND_SIZE, RES_TENSOR);
    """)
    output_size = Var(dtype='int')
    output_size.attr['is_arg'] = False

    output_tensor = Tensor((4096, ), dtype='int')
    output_tensor.attr['is_arg'] = False

    output_size = inline(src, ('RES_SIZE', output_size), \
                                ('FIRST_TENSOR', a[0]), ('FIRST_SIZE', a._size()[0]), \
                                ('SECOND_TENSOR', b[0]), ('SECOND_SIZE',  b._size()[0]),  \
                                ('RES_TENSOR', output_tensor[0]))

    ret_val = output_tensor[0:output_size]
    ret_val.ref_size[0].attr['dynamic_size'] = True
    ret_val.attr['is_set'] = True
    return ret_val


def fill_hash_table(node_list, rowptr, table):
    src = inspect.cleandoc("""
    for(int i=0; i<NODE_LIST_SIZE; i++){
        int vid = NODE_LIST_ARRAY[i];
        int hash_key = Hashing(HASH_TABLE, vid);
        HASH_TABLE[hash_key][0] = NODE_LIST_ARRAY[i];
        HASH_TABLE[hash_key][1] = ROWPTR[vid];
        HASH_TABLE[hash_key][2] = ROWPTR[vid+1];
    }
    """)

    table = inline(src, ('HASH_TABLE', table), \
                        ('NODE_LIST_ARRAY', node_list), ('NODE_LIST_SIZE', node_list._size()[0]), \
                        ('ROWPTR', rowptr))
    return table

def get_hash_key(vid, hash_table):
    src = inspect.cleandoc("""
        KEY = Hashing(HASH_TABLE, VID);
    """)

    key = Var(dtype='int')
    key.attr['is_arg'] = False
    key = inline(src, ('KEY', key), \
                        ('HASH_TABLE', hash_table), ('VID', vid))
    return key

def RapidMatch(pattern_file_name):

    pmtx = read_query_file(pattern_file_name)
    num_query_node, num_query_edge, query_edge_list = query_info(pmtx)
    query_edge_order = query_dfs(pmtx)
    pattern_size = len(pmtx)

    pattern_size = len(pmtx)

    num_node =  Var(name='num_node', dtype='int')
    num_edge =  Var(name='num_edge', dtype='int')
    num_jobs =  Var(name='num_jobs', dtype='int')
    
    node_list = Tensor((num_node,), dtype='int', name='node_list')
    rowptr = Tensor((num_node+1,), dtype='int', name='rowptr')
    colidx = Tensor((num_edge,), dtype='int', name='colidx')
    edge_list =  Tensor((num_jobs, 2), dtype='int', name='edge_list')

    num_core = 3

    relation_dict = {}
    for this_order in query_edge_order:
        this_order.sort()
        u0 = this_order[0]
        u1 = this_order[1]

        hash_table_idx = len(relation_dict)

        if u1>=num_core:
            assert(u0<num_core)
            relation_dict[u1] = Tensor((num_edge, 3), dtype='int', name=('hash_table'+str(hash_table_idx)))
            #relation_dict[u1] =  fill_hash_table(node_list, rowptr, Tensor((num_edge, 3), dtype='int', name=('hash_table'+str(hash_table_idx))))
            # relation_dict[u1].attr['is_arg'] = False


    # for uid in relation_dict.keys():
    #     relation_dict[uid] = fill_hash_table(node_list, rowptr, relation_dict[uid])


    class _RapidMatch:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  colidx[rowptr[v0]:rowptr[v0+1]]
                v1_nb =  colidx[rowptr[v1]:rowptr[v1+1]]
                nb = pmtx[self.level]

                if nb[0]==1 and nb[1]==1:
                    candidate_set = intersect(v0_nb, v1_nb)
                elif nb[1]==1:
                    candidate_set = v1_nb
                elif nb[0]==1:
                    candidate_set = v0_nb
                return candidate_set.apply(_RapidMatch(self.level+1, [v0, v1])).sum()
            elif self.level < num_core:
                all_path = self.path + [item]
                nb = pmtx[self.level]

                first_nb_idx= 0
                for i in range(len(all_path)):
                    if nb[i]!=0:
                        first_nb_idx = i
                        break
                first_node = all_path[first_nb_idx]
                candidate_set = colidx[rowptr[first_node]:rowptr[first_node+1]]

                for i in range(len(all_path)):
                    if nb[i]==1 and i!=first_nb_idx:
                        v =  all_path[i]
                        v_nb =  colidx[rowptr[v]:rowptr[v+1]]
                        candidate_set = intersect(candidate_set, v_nb)

                if self.level == pattern_size-1:
                    return candidate_set.size(0)
                else:
                    return candidate_set.apply(_RapidMatch(self.level+1, self.path + [item])).sum()
            else:
                this_table = relation_dict[self.level]

                all_path = self.path + [item]
                nb = pmtx[self.level]
                for i in range(len(all_path)):
                    if nb[i]==1:
                        v =  all_path[i]
                        pos = get_hash_key(this_table, v)
                        start =  this_table[pos][1]
                        end = this_table[pos][2]
                        candidate_set =  colidx[start:end] #colidx[rowptr[v]:rowptr[v+1]]

                if self.level == pattern_size-1:
                    return candidate_set.size(0)
                else:
                    return candidate_set.apply(_RapidMatch(self.level+1, self.path + [item])).sum()

    
    

    res = edge_list.apply(_RapidMatch(2)).sum()
    
    res = gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

    torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    torch_hash_table = build_hasb_table(torch_rowptr)
    torch.set_printoptions(profile='full')
    d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx, torch_hash_table)
    print(d)

if __name__ == "__main__":
    RapidMatch(sys.argv[1])