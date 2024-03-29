from asg import *
from helpers import flatten, same_object, get_loops_at_level, IRTraversal, replace_all_ref
from asg import *
from ir import *
import codegen
import copy

def _get_level_loops(compute, loop_dict, level):
    lnest = flatten(compute)
    for l in lnest:
        if isinstance(l, Loop):
            if level in loop_dict:
                loop_dict[level].append(l)
            else:
                loop_dict[level] = [l]
            _get_level_loops(l.body, loop_dict, level+1)

def _replace_loop(ir, old, new):
    def action(stmt, res):
        if type(stmt) in (list, tuple):
            if old in stmt:
                idx = stmt.index(old)
                stmt[idx] = new
                return [False]
        return [True, True, True, True]

    t = IRTraversal(action)
    t(ir)

def _if_in_loop(loop, body):
    def action(s, res):
        if isinstance(s, Loop):
            if body in s.body or body == s:
                res.append(body)
        return [True, True, True, True, True]
    t = IRTraversal(action)(loop)
    
    if t:
        return True
    else:
        return False

def _loop_copy(loop):
    # new_loop = Loop(loop.start, loop.end, loop.step, [])
    new_loop = copy.deepcopy(loop)
    new_loop.body = []
    new_loop.attr = loop.attr
    return new_loop

def _recurent_interchange(outer, current_loop, inner, level, t_body):
    if level > 0:
        for i, s in enumerate(current_loop.body):
            if isinstance(s, Assignment):
                # need check if assign has outer.iterate?
                new_loop = _loop_copy(current_loop)
                new_loop.body.append(s)
                # change binded iterate
                new_loop.attr['parent_loop'].body.append(new_loop)
                t_body.append((i, s))
            elif isinstance(s, Loop):
                if _if_in_loop(s, inner):
                    _recurent_interchange(outer, s, inner, level-1, t_body)
                else:
                    new_loop = _loop_copy(current_loop)
                    new_loop.body.append(s)
                    # change binded iterate
                    new_loop.attr['parent_loop'].body.append(new_loop)
                    t_body.append((i, s))

def interchange(node, swap_order):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval', 'norm', 'aggr']
    assert swap_order[0] < swap_order[1]

    loop_dict = {}
    _get_level_loops(node.compute, loop_dict, 0)
    if swap_order[0] in loop_dict and swap_order[1] in loop_dict:
        for outer in loop_dict[swap_order[0]]:
            for inner in loop_dict[swap_order[1]]:
                if _if_in_loop(outer, inner):
                    t_body = []
                    outer.attr['parent_loop'].body = []
                    _recurent_interchange(outer, outer, inner, swap_order[1]-swap_order[0], t_body)
                    
                    raw_num = [i for i in range(len(outer.body))]
                    for i in t_body:
                        raw_num.remove(i[0])
                        outer.body.remove(i[1])
                    remain = outer.body
                    outer.body = inner.body
                    _replace_loop(inner.attr['parent_loop'], inner, outer)
                    if swap_order[1]-swap_order[0] == 1:
                        inner.body = [outer]
                    else:
                        for i in range(swap_order[1]-swap_order[0]-1):
                            remain = remain.body[0]
                        inner.body = remain
                        
                    outer.attr['parent_loop'].body.insert(raw_num[0], inner)


def general_interchange(node, swap_order):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval']
    assert swap_order[0] < swap_order[1]

    loop_dict = {}
    _get_level_loops(node.compute, loop_dict, 0)

    for outer in loop_dict[swap_order[0]]:
        for inner in loop_dict[swap_order[1]]:
            if _if_in_loop(outer, inner):
                temp = inner.body
                ploop = inner.attr['parent_loop']
                inner.body = [outer]
                inner.attr['parent_loop'] = outer.attr['parent_loop']
                outer.body = temp
                outer.attr['parent_loop'] = ploop
                inner.attr['parent_loop'].body = [inner]