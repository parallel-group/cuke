import sys
import numpy as np
import torch

import run
import codegen.cpu
import transform
from asg import *
from asg2ir import gen_ir
from transform import parallelize
from helpers import new_op, ASGTraversal

from apps.gpm.utils import *
from apps.gpm.setop import *


class fuser:
    def __init__(self):
        self.rules = [transform.fuse.basic_rule]

    def __call__(self, node):
        def action(node, res):
            for r in self.rules:
                r(node, res)

        t = ASGTraversal(action)
        t(node)
        return node

class parallelize():
    def __init__(self):
        self.rules = []

    def __call__(self, node):
        transform.parallelize.parallelize_loop(node, 64, [0])
        return node

transform.passes = [fuser(), parallelize()]

def DryadicVertexInduced(pattern_file_name):

    pmtx = read_query_file(pattern_file_name)
    partial_orders = symmetry_breaking(pmtx)
    pattern_size = len(pmtx)

    num_node =  Var(name='num_node', dtype='int')
    num_edge =  Var(name='num_edge', dtype='int')
    num_jobs =  Var(name='num_jobs', dtype='int')
    
    rowptr = Tensor((num_node+1,), dtype='int', name='rowptr')
    colidx = Tensor((num_edge,), dtype='int', name='colidx')
    edge_list =  Tensor((num_jobs, 2), dtype='int', name='edge_list')

    class _DryadicVertexInduced:

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
                elif nb[0]==0 and nb[1]==1:
                    candidate_set = difference(v1_nb, v0_nb)
                elif nb[0]==1 and nb[1]==0:
                    candidate_set = difference(v0_nb, v1_nb)

                return candidate_set.apply(_DryadicVertexInduced(self.level+1, [v0, v1])).sum()
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
                    return candidate_set.apply(_DryadicVertexInduced(self.level+1, self.path + [item])).sum()
    
    

    res = edge_list.apply(_DryadicVertexInduced(2)).sum()
    
    res = gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

    torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    print(d)


def DryadicEdgeInduced(pattern_file_name):

    pmtx = read_query_file(pattern_file_name)
    partial_orders = symmetry_breaking(pmtx)
    pattern_size = len(pmtx)

    num_node =  Var(name='num_node', dtype='int32_t')
    num_edge =  Var(name='num_edge', dtype='int64_t')
    num_jobs =  Var(name='num_jobs', dtype='int64_t')
    
    rowptr = Tensor((num_node,), dtype='int64_t', name='rowptr')
    colidx = Tensor((num_edge,), dtype='int32_t', name='colidx')
    edge_list =  Tensor((num_jobs, 2), dtype='int32_t', name='edge_list')

    class _DryadicEdgeInduced:

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
                elif nb[0]==0 and nb[1]==1:
                    candidate_set = v1_nb

                elif nb[0]==1 and nb[1]==0:
                    candidate_set = v0_nb

                # candidate_set.attr['is_set'] = True
                if self.level == pattern_size-1:
                    return candidate_set.size(0)
                else:
                    return candidate_set.apply(_DryadicEdgeInduced(self.level+1, [v0, v1])).sum()
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

                # candidate_set.attr['is_set'] = True
                if self.level == pattern_size-1:
                    return candidate_set.size(0)
                else:
                    return candidate_set.apply(_DryadicEdgeInduced(self.level+1, self.path + [item])).sum()
    

    res = edge_list.apply(_DryadicEdgeInduced(2)).sum()
    
    res = gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

    # torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    # d = run.cpu.compile_and_run(code, num_jobs, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    # print(d)

if __name__ == "__main__":
    DryadicEdgeInduced(sys.argv[1])
    # DryadicVertexInduced(sys.argv[1])