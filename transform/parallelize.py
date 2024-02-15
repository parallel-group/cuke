import transform
from helpers import ASGTraversal, IRTraversal, get_obj, replace_all_ref, same_object, flatten, get_loops_at_level
from asg import *
from ir import *
import codegen
from asg2ir import gen_ir

def _get_ref_idx(stmt, v):
    def _get_scalar_idx(idx):
        assert type(idx) == Indexing

        def action(s, res):
            if type(s) in (Scalar, Literal):
                res.append(s)
            elif type(s) == Ndarray:
                return [False]

            return [True, True, True, True, True]

        t = IRTraversal(action)
        res = t(idx)
        return res

    def action(s, res):
        if type(s) == Indexing:
            if get_obj(s).dobject_id == v.dobject_id:
                if len(res) == 0:
                    res.append(_get_scalar_idx(s))
                return [False, False]
            else:
                return [False, True]
        elif type(s) in (Scalar, Ndarray):
            if s.dobject_id == v.dobject_id:
                if len(res) == 0:
                    res.append([])

        return [True, True, True, True, True]

    t = IRTraversal(action)
    res = t(stmt)
    if len(res) > 0:
        return res[0]
    else:
        return None

def _replace_all_ref(stmt, old, new):
    def action(s, res):
        match s.__class__.__name__:
            case 'Loop':
                if same_object(s.start, old):
                    s.start = new
                if same_object(s.end, old):
                    s.end = new
                if same_object(s.step, old):
                    s.step = new
            case 'FilterLoop':
                if same_object(s.cond, old):
                    s.cond = new
                if same_object(s.start, old):
                    s.start = new
                if same_object(s.end, old):
                    s.end = new
                if same_object(s.step, old):
                    s.step = new
            case 'Expr':
                if same_object(s.left, old):
                    s.left = new
                if same_object(s.right, old):
                    s.right = new
            case 'Assignment':
                if same_object(s.lhs, old):
                    s.lhs = new
                if same_object(s.rhs, old):
                    s.rhs = new
            case 'Indexing':
                if same_object(s.dobject, old):
                    temp = s.dobject
                    idx_list =[s.idx]
                    while isinstance(temp, Indexing):
                        idx_list.append(temp.idx)
                        temp = temp.doibject
                    idx_list.reverse()
                    s.idx = new.idx
                    new_obj = new.dobject
                    for i in idx_list:
                        new_obj = Indexing(new_obj, i)
                    s.dobject = new_obj
                if same_object(s.idx, old):
                    s.idx = new
            case 'Slice':
                if same_object(s.start, old):
                    s.start = new
                if same_object(s.stop, old):
                    s.stop = new
                if same_object(s.step, old):
                    s.step = new
            case 'Math':
                if same_object(s.val, old):
                    s.val = new
            case 'Code':
                if same_object(s.output[1], old):
                    s.output = (s.output[0], new)
                for k in s.inputs:
                    if s.inputs[k] == old:
                        s.inputs[k] = new
        return [True, True, True, True, True]

    t = IRTraversal(action)
    res = t(stmt)

def parallelize_loop(node, num_procs, idx: list | tuple):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval']
    assert type(num_procs) == int and num_procs > 0

    scope = flatten(node.compute)
    nprocs = []
    for i in idx:
        loop = None
        j = -1
        for s in scope:
            if 'reduction' in s.attr:
                continue
            if isinstance(s, Loop):
                j += 1
                if j == i:
                    loop = s
                    break
        if loop != None:
            if 'nprocs' in loop.attr:
                nprocs = loop.attr['nprocs']
            scope = (flatten(loop.cond_body) if type(loop) == FilterLoop else []) + flatten(loop.body)

    if loop != None:
        loop.attr['plevel'] = len(nprocs)
        loop.attr['nprocs'] = nprocs + [(num_procs, loop)]

        def set_nprocs(s, res):
            if type(s) in (list, tuple):
                return [True]
            if isinstance(s, Loop):
                for ss in flatten(s.body):
                    if isinstance(ss, Loop):
                        ss.attr['nprocs'] = loop.attr['nprocs'].copy()
                if type(s) == FilterLoop:
                    for ss in flatten(s.cond_body):
                        if isinstance(ss, Loop):
                            ss.attr['nprocs'] = loop.attr['nprocs'].copy()
                return [False, False, False, True, True]

            return [False, False, False, False, False]

        IRTraversal(set_nprocs)(loop)

        def get_assigns(s, res):
            if type(s) in (Assignment, Code):
                res.append(s)
            return [True, True, True, True, True]

        assigns = IRTraversal(get_assigns)(loop)
        # assigns = assigns[:-1]
        def get_vars(n, res):
            res.extend([d.dobject for d in n.decl])

        all_vars = ASGTraversal(get_vars)(node)
        to_replace = {}
        for s in assigns:
            for v in all_vars:
                if v not in to_replace:
                    lhs = s.lhs if type(s) == Assignment else s.output
                    idx = _get_ref_idx(lhs, v)
                    if idx != None:
                        ext_size = []
                        loop_info = []
                        for l in loop.attr['nprocs']:
                            indexed = False
                            for ii in idx:
                                if ('loop' in ii.attr and ('output_axis' in l[1].attr and l[1].attr['output_axis'] ==
                                                           ii.attr['loop'].attr['output_axis']) or same_object(ii, l[
                                    1].iterate))  or ('ploop' in ii.attr and ii.attr['ploop'] == l[1]):
                                    indexed = True
                                    break
                            if not indexed:
                                ext_size.append(l[0])
                                loop_info.append(l[1])
                        
                        if len(ext_size) > 0:
                            to_replace[v] = (Ndarray(v.dtype, ext_size + v.size), loop_info)
        
        # for key in to_replace:
        #     print('-->', key, to_replace[key], to_replace[key][0].size, to_replace[key][1][0].attr, codegen.gpu.to_string(key), codegen.gpu.to_string(to_replace[key]))

        def replace_decls(n, res):
            decl = []
            for d in n.decl:
                v = d.dobject
                replace_with = None
                for k in to_replace:
                    if same_object(k, v):
                        replace_with = to_replace[k][0]
                        break
                if replace_with != None:
                    decl.append(Decl(replace_with))
                    if same_object(v, n.eval):
                        n.eval = replace_with
                else:
                    decl.append(d)
            n.decl = decl

        ASGTraversal(replace_decls)(node)

        def replace_refs(n, res):
            if len(n.compute) > 0:
                for s in to_replace:
                    new_var = to_replace[s][0]
                    for l in to_replace[s][1]:
                        idx = Literal(f"tid{l.attr['plevel']}", 'int')
                        idx.attr['ploop'] = l
                        new_var = Indexing(new_var, idx)
                    replace_all_ref(n.compute, s, new_var)

        ASGTraversal(replace_refs)(node)

        def add_reduction_spec(s, res):
            def _is_in_loopbody(loop, body, index):
                for i, element in enumerate(body):
                    if isinstance(element, list|tuple):
                        index.append(i)
                        if _is_in_loopbody(loop, element, index):
                            return True
                        index.pop()
                    elif element == loop:
                        index.append(i)
                        return True
                return False
            
            if isinstance(s, Loop) and 'ptype' in s.attr and s.attr['ptype'] == 'reduction' and 'plevel' in s.attr:
                redu_eval = None
                for ass in s.body:
                    if type(ass) == Assignment:
                        redu_eval = ass.lhs 
                    else:
                        continue
                if isinstance(redu_eval, (Ndarray, Indexing)):
                    new_res = Scalar(redu_eval.dtype)
                    s.attr['redu_res'] = new_res
                    replace_all_ref(s, redu_eval, new_res)
                
                    s.attr['redu_eval'] = redu_eval
                    if len(s.attr['nprocs']) > 1:
                        outerloop = s.attr['nprocs'][-2][1]
                    else:
                        outerloop = s.attr['nprocs'][-1][1]
                    
                    redu_loop = Loop(0, s.attr['nprocs'][-1][0], 1, [])
                    redu_loop.body.append(Assignment(redu_eval, new_res, '+'))
                    redu_loop.attr['reduction'] = True
                    redu_assign = Assignment(redu_eval, new_res)
                    
                    pos = []
                    _is_in_loopbody(s.attr['parent_loop'], outerloop.body, pos)
                    temp = outerloop.body

                    if len(pos) > 0:
                        for i in range(len(pos)-1):
                            temp = temp[pos[i]]
                        temp.insert(pos[-1]+1, redu_assign)
                        temp.insert(pos[-1]+1, redu_loop)
                
            return [True, True, True, True, True]
        
        
        IRTraversal(add_reduction_spec)(loop)

def parallelize_level(node, num_procs, level):
    loops = []
    get_loops_at_level(node.compute, level, [], loops)
    for l in loops:
        parallelize_loop(node, num_procs, l)

# class parallelizer:
#
#     def __init__(self, num_procs=[16]):
#         self.num_procs = num_procs
#
#     def __call__(self, node):
#
#         def find_all_vars(n, res):
#             res.extend([d.dobject for d in n.decl])
#
#         all_vars = ASGTraversal(find_all_vars)(node)
#
#         def ir_set_loop_level(stmt, num_procs):
#             def action(s, res):
#                 if type(s) in (list, tuple):
#                     return [True]
#                 if isinstance(s, Loop):
#                     if not 'plevel' in s.attr and not 'nprocs' in s.attr:
#                         s.attr['plevel'] = 0
#                         s.attr['nprocs'] = [(num_procs[0], s)]
#                     for ss in flatten(s.body):
#                         if isinstance(ss, Loop):
#                             if 'plevel' in s.attr and s.attr['plevel'] + 1 < len(num_procs):
#                                 ss.attr['plevel'] = s.attr['plevel'] + 1
#                                 ss.attr['nprocs'] = s.attr['nprocs'] + [(num_procs[ss.attr['plevel']], ss)]
#                             else:
#                                 ss.attr['nprocs'] = s.attr['nprocs']
#                     if type(s) == FilterLoop:
#                         for ss in flatten(s.cond_body):
#                             if isinstance(ss, Loop):
#                                 if 'plevel' in s.attr and s.attr['plevel'] + 1 < len(num_procs):
#                                     ss.attr['plevel'] = s.attr['plevel'] + 1
#                                     ss.attr['nprocs'] = s.attr['nprocs'] + [(num_procs[ss.attr['plevel']], ss)]
#                                 else:
#                                     ss.attr['nprocs'] = s.attr['nprocs']
#                     return [False, False, False, True, True]
#
#                 return [False, False, False, False, False]
#
#             IRTraversal(action)(stmt)
#
#         def ir_find_data_races(stmt, to_replace):
#             def action(s, res):
#                 if isinstance(s, Loop):
#                     if 'nprocs' in s.attr:
#                         assigns = [ss for ss in flatten(s.body) if type(ss) in (Assignment, Code)]
#                         if type(s) == FilterLoop:
#                             assigns.extend([ss for ss in flatten(s.cond_body) if type(ss) in (Assignment, Code)])
#                         for ss in assigns:
#                             for v in all_vars:
#                                 if v not in to_replace:
#                                     lhs = ss.lhs if type(ss) == Assignment else ss.output
#                                     idx = _get_ref_idx(lhs, v)
#                                     if idx != None:
#                                         ext_size = []
#                                         loop_info = []
#                                         for l in s.attr['nprocs']:
#                                             indexed = False
#                                             for ii in idx:
#                                                 if ('output_axis' in l[1].attr and l[1].attr['output_axis'] ==
#                                                     ii.attr['loop'].attr['output_axis']) or same_object(ii, l[
#                                                     1].iterate):
#                                                     indexed = True
#                                                     break
#                                             if not indexed:
#                                                 ext_size.append(l[0])
#                                                 loop_info.append(l[1])
#                                         if len(ext_size) > 0:
#                                             to_replace[v] = (Ndarray(v.dtype, ext_size + v.size), loop_info)
#
#                 return [True, True, True, True, True]
#
#             IRTraversal(action)(stmt)
#
#
#         def find_shared_vars(n, res):
#             if len(res) == 0:
#                 res.append({})
#             if not 'scope' in n.attr and len(n.compute) != 0:
#                 ir_set_loop_level(n.compute, self.num_procs)
#                 ir_find_data_races(n.compute, res[0])
#
#         to_replace = ASGTraversal(find_shared_vars)(node)[0]
#
#         def replace_decls(n, res):
#             decl = []
#             for d in n.decl:
#                 v = d.dobject
#                 replace_with = None
#                 for k in to_replace:
#                     if same_object(k, v):
#                         replace_with = to_replace[k][0]
#                         break
#                 if replace_with != None:
#                     decl.append(Decl(replace_with))
#                     if same_object(v, n.eval):
#                         n.eval = replace_with
#                 else:
#                     decl.append(d)
#             n.decl = decl
#
#         ASGTraversal(replace_decls)(node)
#
#         def replace_refs(n, res):
#             if len(n.compute) > 0:
#                 for s in to_replace:
#                     new_var = to_replace[s][0]
#                     for l in to_replace[s][1]:
#                         new_var = Indexing(new_var, Literal(f"tid{l.attr['plevel']}", 'int'))
#                     replace_all_ref(n.compute, s, new_var)
#
#         ASGTraversal(replace_refs)(node)
#         return node


def test1():
    A = Tensor((10, 20))
    B = Tensor((10, 20))
    C = Tensor((10, 20))
    D = Tensor((10, 20))

    A = setval(A, 1)
    t1 = A + B
    t2 = (C - D).abs()
    res1 = t1 + t2
    ir = gen_ir(res1)
    parallelize_loop(t1, 16, [0])
    code = codegen.cpu.print_cpp(ir)
    print(code)


def test2():
    A = Tensor((10, 20))
    B = Tensor((20, 30))
    C = Tensor((10, 20))
    D = Tensor((20, 30))
    t1 = A @ B
    t2 = (C @ D).round()
    res1 = t1 + t2
    ir = gen_ir(res1)
    parallelize_loop(t1, 16, [0])
    print(codegen.cpu.print_cpp(ir))


if __name__ == "__main__":
    # test1()
    test2()
