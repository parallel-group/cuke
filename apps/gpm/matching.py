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

def symmetry_breaking(pmtx):
    num_node = len(pmtx)
    perms = list(permutations(list(range(num_node))))

    valid_perms = []
    for perm in perms:
        mapped_adj = [None for x in range(num_node)]
        for i in range(num_node):
            tp = set()
            for j in range(num_node):
                if pmtx[i][j]==0:
                    continue
                tp.add(perm[j])
            mapped_adj[perm[i]] = tp

        valid = True
        for i in range(num_node):
            equal = True
            count = 0
            for j in range(num_node):
                if pmtx[i][j]==1:
                    count+=1
                    if j not in mapped_adj[i]:
                        equal=False
            
            if not equal or count!=len(mapped_adj[i]):
                valid = False
                break

        if valid==True:
            valid_perms.append(perm)

    partial_orders = [[0 for y in range(num_node)] for x in range(num_node)]
    for i in range(num_node):
        stabilized_aut = []
        for perm in valid_perms:
            if perm[i]==i:
                stabilized_aut.append(perm)
            else:
                partial_orders[perm[i]][i]=1

        valid_perms = stabilized_aut
    

    res = [-1 for x in range(num_node)]
    for i in range(num_node):
        largest_idx = -1
        for j in range(num_node):
            if partial_orders[i][j]==1 and j>largest_idx:
                largest_idx = j
        
        res[i] = largest_idx
    print("partial orders:")
    print(res)
    return res

def read_pattern_file(filename):
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
        print("pmtx:")
        print(pmtx)
        return pmtx

def node_degree(pmtx, idx):
    res = 0
    row = pmtx[idx]
    for i in row:
        if i==1:
            res+=1
    return res

def is_in(x, li):
    src = inspect.cleandoc("""
    F = BinarySearch(LI, 0, LSIZE, X);
    """)
    found = Var(dtype='int')
    found.attr['is_arg'] = False
    return inline(src, ('F', found), ('X', x), ('LI', li[0]), ('LSIZE', li._size()[0]))

def is_not_in(x, li):
    src = inspect.cleandoc("""
    F = !BinarySearch(LI, 0, LSIZE, X);
    """)
    found = Var(dtype='int')
    found.attr['is_arg'] = False
    return inline(src, ('F', found), ('X', x), ('LI', li[0]), ('LSIZE', li._size()[0]))

def intersect(a, b):
    c = a.apply(lambda x: is_in(x, b))
    return a.apply(lambda x: x, cond=c)

def difference(a, b):
    c = a.apply(lambda x: is_not_in(x, b))
    return a.apply(lambda x: x, cond=c)


def SubgraphMatchingVertexInduced(pattern_file_name):

    pmtx = read_pattern_file(pattern_file_name)
    partial_orders = symmetry_breaking(pmtx)
    pattern_size = len(pmtx)

    num_node =  Var(name='num_node', dtype='int')
    num_edge =  Var(name='num_edge', dtype='int')
    num_jobs =  Var(name='num_jobs', dtype='int')
    
    rowptr = Tensor((num_node+1,), dtype='int', name='rowptr')
    colidx = Tensor((num_edge,), dtype='int', name='colidx')
    edge_list =  Tensor((num_jobs, 2), dtype='int', name='edge_list')
    count = Var(name='count', dtype='int')
    count.attr['is_arg'] = False

    class _SubgraphMatchingVertexInduced:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            # if self.level == pattern_size:
            #     return setval(count, 1)
            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  colidx[rowptr[v0]:rowptr[v0+1]]
                v1_nb =  colidx[rowptr[v1]:rowptr[v1+1]]
                nb = pmtx[self.level]

                if nb[0]==1 and nb[1]==1:
                    candidate_set = intersect(v0_nb, v1_nb)
                elif nb[0]==0 and nb[1]==1:
                    candidate_set = difference(v1_nb, v0_nb)
                elif nb[0]==1 and nb[1]==0:
                    candidate_set = difference(v0_nb, v1_nb)

                return candidate_set.apply(_SubgraphMatchingVertexInduced(self.level+1, [v0, v1])).sum()
            else:
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
                    if i==first_nb_idx:
                        continue
                    v =  all_path[i]
                    v_nb =  colidx[rowptr[v]:rowptr[v+1]]
                    if nb[i]==1:
                        candidate_set = intersect(candidate_set, v_nb)
                    else:
                        candidate_set = difference(candidate_set, v_nb)

                if self.level == pattern_size-1:
                    return candidate_set.size(0)
                else:
                    return candidate_set.apply(_SubgraphMatchingVertexInduced(self.level+1, self.path + [item])).sum()
    
    

    res = edge_list.apply(_SubgraphMatchingVertexInduced(2)).sum()
    
    res = gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

    torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    print(d)

if __name__ == "__main__":
    SubgraphMatchingVertexInduced(sys.argv[1])