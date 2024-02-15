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