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

def is_in(x, li):
    src = inspect.cleandoc("""
    F = BinarySearch(LI, 0, LSIZE, X);
    """)
    found = Var(dtype='int')
    found.attr['is_arg'] = False
    return inline(src, ('F', found), ('X', x), ('LI', li[0]), ('LSIZE', li._size()[0]))

def intersect(a, b):
    c = a.apply(lambda x: is_in(x, b))
    return a.apply(lambda x: x, cond=c)

def p1_decomine():

    num_node =  Var(name='num_node', dtype='int')
    num_edge =  Var(name='num_edge', dtype='int')
    num_jobs =  Var(name='num_jobs', dtype='int')
    
    rowptr = Tensor((num_node+1,), dtype='int', name='rowptr')
    colidx = Tensor((num_edge,), dtype='int', name='colidx')
    edge_list =  Tensor((num_jobs, 2), dtype='int', name='edge_list')

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  colidx[rowptr[v0]:rowptr[v0+1]]
                v1_nb =  colidx[rowptr[v1]:rowptr[v1+1]]

                candidate_set = intersect(v0_nb, v1_nb)
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1])).sum()
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb =  colidx[rowptr[v0]:rowptr[v0+1]]
                v1_nb =  colidx[rowptr[v1]:rowptr[v1+1]]
                v2_nb =  colidx[rowptr[v2]:rowptr[v2+1]]
                
                candidate_v3 = intersect(v0_nb, v1_nb)
                candidate_v4 = intersect(v0_nb, v2_nb)
                candidate_v5 = intersect(v1_nb, v2_nb)

                candidate_shrink =  intersect(candidate_v3, v2_nb)

                return candidate_v3.size(0) * candidate_v4.size(0) * candidate_v5.size(0) - candidate_shrink.size(0)*3         

    res = edge_list.apply(inner_subgraph_matching(2)).sum()
    gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)
    torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    print(d)


def p2_decomine():
    num_node =  Var(name='num_node', dtype='int')
    num_edge =  Var(name='num_edge', dtype='int')
    num_jobs =  Var(name='num_jobs', dtype='int')
    
    rowptr = Tensor((num_node+1,), dtype='int', name='rowptr')
    colidx = Tensor((num_edge,), dtype='int', name='colidx')
    edge_list =  Tensor((num_jobs, 2), dtype='int', name='edge_list')

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  colidx[rowptr[v0]:rowptr[v0+1]]
                v1_nb =  colidx[rowptr[v1]:rowptr[v1+1]]

                candidate_set = intersect(v0_nb, v1_nb)
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1])).sum()
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb =  colidx[rowptr[v0]:rowptr[v0+1]]
                v1_nb =  colidx[rowptr[v1]:rowptr[v1+1]]
                v2_nb =  colidx[rowptr[v2]:rowptr[v2+1]]

                candidate_v5 = intersect(v1_nb, v2_nb)

                candidate_v3v4 = intersect(candidate_v5, v0_nb)

                v5_nelem = candidate_v5.size(0)
                v3v4_nelem = candidate_v3v4.size(0)

                return v5_nelem * v3v4_nelem * v3v4_nelem - v3v4_nelem

    res = edge_list.apply(inner_subgraph_matching(2)).sum()
    
    gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)
    torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    print(d)



def p3_decomine():
    num_node =  Var(name='num_node', dtype='int')
    num_edge =  Var(name='num_edge', dtype='int')
    num_jobs =  Var(name='num_jobs', dtype='int')
    
    rowptr = Tensor((num_node+1,), dtype='int', name='rowptr')
    colidx = Tensor((num_edge,), dtype='int', name='colidx')
    edge_list =  Tensor((num_jobs, 2), dtype='int', name='edge_list')

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v1_nb =  colidx[rowptr[v1]:rowptr[v1+1]]

                candidate_v2 = v1_nb
                return candidate_v2.apply(inner_subgraph_matching(self.level+1, [v0, v1])).sum()
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb = colidx[rowptr[v0]:rowptr[v0+1]]
                v1_nb = colidx[rowptr[v1]:rowptr[v1+1]]
                v2_nb = colidx[rowptr[v2]:rowptr[v2+1]]

                candidate_v3v4 = intersect(v0_nb, v1_nb)
                candidate_v5 = intersect(v0_nb, v2_nb)
                
                v5_nelem = candidate_v5.size(0)
                v3v4_nelem = candidate_v3v4.size(0)

                return v5_nelem * v3v4_nelem * v3v4_nelem - v3v4_nelem                
    
    res = edge_list.apply(inner_subgraph_matching(2)).sum()
    gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)
    torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    print(d)


if __name__ == "__main__":
    # p1_decomine()
    # p2_decomine()
    p3_decomine()