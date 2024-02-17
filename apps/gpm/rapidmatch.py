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

from apps.gpm.utils import *
from apps.gpm.setop import *

def read_rapidmatch_query_file(filename):
    pmtx = None
    num_node=0
    num_core=-1
    plabel = []
    with open(filename) as p:
        for line in p:
            if line[0]=='v':
                num_node+=1
                plabel.append(int(line[4]))
            if line[0]=='c':
                num_core=int(line[2])
        p.seek(0, 0)
        
        if num_core==-1:
            num_core=num_node

        pmtx = [[0 for y in range(num_node)] for x in range(num_node)]

        for line in p: 
            if line[0]=='e':
                v0 = int(line[2])
                v1 = int(line[4])
                pmtx[v0][v1] = 1
                pmtx[v1][v0] = 1
        return pmtx, plabel, num_core

def compute_hash_key(hash_table, node_id):
    key = node_id*2
    if node_id!=0 and hash_table[key][0]!=0:
        while key < hash_table.shape[0]:
            if hash_table[key][0]!=0:
                key+=1
    hash_table[key][0] = node_id
    return key


def build_hasb_table(torch_rowptr, torch_label, torch_colidx, end_node_label):
    num_vertex = torch_rowptr.shape[0]-1
    num_edge = torch_colidx.shape[0]

    hash_table = torch.zeros([num_vertex*2, 3], dtype=torch.int32)
    new_colidx = torch.zeros([num_edge, ], dtype=torch.int32)

    new_start = 0
    new_end = 0
    for node_id in range(0, num_vertex):
        old_start = torch_rowptr[node_id]
        old_end =  torch_rowptr[node_id+1]

        for colidx_idx in range(old_start, old_end):
            vid = torch_colidx[colidx_idx]
            if torch_label[vid]==end_node_label:
                torch_colidx[new_end]=vid
                new_end+=1
        
        key = compute_hash_key(hash_table, node_id)
        hash_table[key][0] = node_id
        hash_table[key][1] = new_start
        hash_table[key][2] = new_end
        new_start = new_end

    return hash_table, new_colidx

def get_hash_key(vid, hash_table):
    src = inspect.cleandoc("""
        KEY = Hashing(HASH_TABLE, VID);
    """)

    key = Var(dtype='int')
    key.attr['is_arg'] = False
    key = inline(src, [('KEY', key)], \
                        [('HASH_TABLE', hash_table), ('VID', vid)])
    return key

def RapidMatch(pattern_file_name):

    pmtx, plabel, num_core = read_rapidmatch_query_file(pattern_file_name)
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

    relation_dict = {}
    for this_order in query_edge_order:
        this_order.sort()
        u0 = this_order[0]
        u1 = this_order[1]

        if u1>=num_core:
            assert(u0<num_core)
            this_hash_table = Tensor((num_edge, 3), dtype='int', name=('hash_table'+str(len(relation_dict))))
            this_colidx =  Tensor((num_edge,), dtype='int', name='colidx'+str(len(relation_dict)))
            relation_dict[u1] = [this_hash_table, this_colidx]
            #relation_dict[u1] =  fill_hash_table(node_list, rowptr, Tensor((num_edge, 3), dtype='int', name=('hash_table'+str(hash_table_idx))))
            # relation_dict[u1].attr['is_arg'] = False
    
    num_tail_node = num_query_node-num_core
    assert(len(relation_dict)==num_tail_node)
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
                this_table = relation_dict[self.level][0]
                this_colidx = relation_dict[self.level][1]

                all_path = self.path + [item]
                nb = pmtx[self.level]
                for i in range(len(all_path)):
                    if nb[i]==1:
                        v =  all_path[i]
                        pos = get_hash_key(this_table, v)
                        start =  this_table[pos][1]
                        end = this_table[pos][2]
                        candidate_set =  this_colidx[start:end] #colidx[rowptr[v]:rowptr[v+1]]

                if self.level == pattern_size-1:
                    return candidate_set.size(0)
                else:
                    return candidate_set.apply(_RapidMatch(self.level+1, self.path + [item])).sum()

    
    

    res = edge_list.apply(_RapidMatch(2)).sum()
    
    res = gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)


    # torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)

    # for i in range(num_core, num_query_node):
    #     torch_label = torch.zeros([num_node, ], dtype=torch.int32)
    #     torch_hash_table, torch_this_colidx= build_hasb_table(torch_rowptr, torch_label ,torch_colidx, plabel[i])
    # d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx, torch_hash_table, torch_this_colidx)
    # print(d)

if __name__ == "__main__":
    RapidMatch(sys.argv[1])