import sys
from itertools import permutations 
import numpy as np
import torch

# import run
# import transform
import codegen.cpu
from asg import *
from asg2ir import gen_ir

from apps.gpm.utils import *
from apps.gpm.setop import *


def loop_cond(detected_nodes, equal_node, cur_vertex):
    code = "RES = "

    idx = 0
    param = []

    v_tmp = Var()
    v_tmp.attr['is_arg'] = False
    v_tmp = setval(v_tmp, cur_vertex)

    for u, v in detected_nodes.items():
        if u==equal_node:
            code += f"(X{idx}==Y)"
        else:
            code += f"(X{idx}!=Y)"
        if idx < len(detected_nodes)-1:
            code += "&&"
        else:
            code += ";"
        param.append((f"X{idx}", v))
        idx+=1
    
    param.append(("Y", v_tmp))
    res = Var(dtype='int')
    res.attr['is_arg'] = False
    return inline(code, ('RES', res), *param)


def everest(pattern_file_name):
    pmtx = read_query_file(pattern_file_name)
    num_query_node, num_query_edge, query_edge_list = query_info(pmtx)
    query_edge_order = query_dfs(pmtx)

    print(query_edge_order)

    num_graph_node =  Var(name='num_node', dtype='int')
    num_graph_edge =  Var(name='num_edge', dtype='int')
    
    rowptr = Tensor((num_graph_node+1,), dtype='int', name='rowptr')
    colidx = Tensor((num_graph_edge,), dtype='int', name='colidx')
    edge_list =  Tensor((num_graph_edge, 2), dtype='int', name='edge_list')

    count = Var(name='count', dtype='int')
    count.attr['is_arg'] = False

    # time = Var(name='time_accumulate', dtype='int')
    # time.attr['is_arg'] = False

    time_stamp = []
    detected_nodes = {}
    class _everest:

        def __init__(self, level):
             self.level = level
             self.time_length = 0

        def __call__(self, item):           
            cur_u0 = query_edge_order[self.level][0]
            cur_u1 = query_edge_order[self.level][1]
            # cur_t = time_stamp[self.level]

            if cur_u0 not in detected_nodes:
                cur_v0 = Var()
                cur_v0.attr['is_arg'] = False
                cur_v0 = setval(cur_v0, item[0])
                detected_nodes[cur_u0] = cur_v0

            if cur_u1 not in detected_nodes:
                cur_v1 = Var()
                cur_v1.attr['is_arg'] = False
                cur_v1 = setval(cur_v1, item[1])
                detected_nodes[cur_u1] = cur_v1
            
            #cur_tt = item[2]

            next_u0 = query_edge_order[self.level+1][0]
            next_u1 = query_edge_order[self.level+1][1]
            # next_t = time_stamp[self.level+1]

            assert(next_u0 in detected_nodes)
            next_v0 = detected_nodes[next_u0]

            candidate_edges = edge_list[rowptr[next_v0]:rowptr[next_v0+1]]
            apply_cond = candidate_edges.apply(lambda edge: loop_cond(detected_nodes, next_u1, edge[1]))

            if self.level==num_query_edge-2:
                return candidate_edges.apply(func=lambda x: x, cond=apply_cond).size(0)
            else:
                return candidate_edges.apply(func=_everest(self.level+1), cond=apply_cond).sum()


    u0 = query_edge_order[0][0]
    res = edge_list.apply(_everest(0)).sum()
    
    gen_ir(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

if __name__ == "__main__":
    everest(sys.argv[1])


