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

# def _same_object(a, b):
#     if isinstance(a, DObject) and isinstance(b, DObject):
#         return get_obj(a).dobject_id == get_obj(b).dobject_id
#     return False

def _same_object(a, b):
    if type(a) != type(b):
        return False
    elif type(a)==Indexing and type(b)==Indexing:
        return _same_object(a.dobject, b.dobject) and _same_object(a.idx, b.idx)
    elif type(a)==Slice and type(b)==Slice:
        return same_object(a.start, b.start) and same_object(a.end, b.end) and same_object(a.step, b.step)
    elif type(a)==Scalar and type(b)==Scalar:
        return a.name()==b.name()
    elif type(a)==Ndarray and type(b)==Ndarray:
        return a.name()==b.name()
    else:    
        return same_object(a, b)

def _replace_all_ref(stmt, old, new):
    def action(s, res):
        match s.__class__.__name__:
            case 'Loop':
                if _same_object(s.start, old):
                    s.start = new
                if _same_object(s.end, old):
                    s.end = new
                if _same_object(s.step, old):
                    s.step = new
            case 'FilterLoop':
                if _same_object(s.cond, old):
                    s.cond = new
                if _same_object(s.start, old):
                    s.start = new
                if _same_object(s.end, old):
                    s.end = new
                if _same_object(s.step, old):
                    s.step = new
            case 'Expr':
                if _same_object(s.left, old):
                    s.left = new
                if _same_object(s.right, old):
                    s.right = new
            case 'Assignment':
                if _same_object(s.lhs, old):
                    s.lhs = new
                if _same_object(s.rhs, old):
                    s.rhs = new
            case 'Indexing':
                # if same_object(s.dobject, old):
                #     temp = s.dobject
                #     idx_list =[s.idx]
                #     while isinstance(temp, Indexing):
                #         idx_list.append(temp.idx)
                #         temp = temp.doibject
                #     idx_list.reverse()
                #     s.idx = new.idx
                #     new_obj = new.dobject
                #     for i in idx_list:
                #         new_obj = Indexing(new_obj, i)
                #     s.dobject = new_obj
                # if same_object(s.idx, old):
                #     s.idx = new
                if _same_object(s.dobject, old):
                    s.dobject = new
                if _same_object(s.idx, old):
                    s.idx = new
            case 'Slice':
                if _same_object(s.start, old):
                    s.start = new
                if _same_object(s.stop, old):
                    s.stop = new
                if _same_object(s.step, old):
                    s.step = new
            case 'Math':
                if isinstance(s.val, (list, tuple)):
                    for i in range(len(s.val)):
                        if _same_object(s.val[i], old):
                            s.val[i] = new
                elif _same_object(s.val, old):
                    s.val = new
                # if _same_object(s.val, old):
                #     s.val = new
            case 'Code':
                for k in s.outputs:
                    if _same_object(s.outputs[k], old):
                        s.outputs[k] = new
                for k in s.inputs:
                    if _same_object(s.inputs[k], old):
                        s.inputs[k] = new
        return [True, True, True, True, True]

    t = IRTraversal(action)
    res = t(stmt)

def parallelize_loop(node, num_procs, idx: list | tuple):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval', 'norm', 'aggr']
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

        def get_vars(n, res):
            res.extend([d.dobject for d in n.decl])
        all_vars = ASGTraversal(get_vars)(node)

        to_replace = {}
        for s in assigns:
            for v in all_vars:
                # if v not in to_replace and not same_object(v, node.eval):
                if v not in to_replace:
                    if type(s) == Assignment:
                        lhs_list = [s.lhs]
                    elif type(s) ==Code:
                        lhs_list = list(s.outputs.values())
                    
                    for lhs in lhs_list:
                        idx = _get_ref_idx(lhs, v)
                        if idx != None:                            
                            ext_size = []
                            loop_info = []
                            for l in loop.attr['nprocs']:
                                indexed = False
                                for ii in idx:
                                    if ('loop' in ii.attr and ('output_axis' in l[1].attr and l[1].attr['output_axis'] == ii.attr['loop'].attr['output_axis']) or same_object(ii, l[1].iterate)):
                                        indexed = True
                                        break
                                if not indexed:
                                    ext_size.append(l[0])
                                    loop_info.append(l[1])
                                   
                            if len(ext_size) > 0:
                                num_ext = len(ext_size)
                                arr = Ndarray(v.dtype, ext_size + v.size[num_ext-1:])
                                to_replace[v] = (arr, loop_info)
        
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
                    replace_with.attr['mem_layer'] = 'global'
                    e = n.eval
                    e.attr['storage'].append(replace_with)
                    # while 'cache' in e.attr :
                    #     e = e.attr['cache']
                    e.attr['cache'] = replace_with
                else:
                    decl.append(d)
            n.decl = decl

        ASGTraversal(replace_decls)(node)

        def replace_refs(n, res):
            if len(n.compute) > 0:
                for s in to_replace:

                    old_var = s
                    new_var = to_replace[s][0]
                    for i in range(len(to_replace[s][1])):
                        l = to_replace[s][1][i]
                        idx = Scalar('int', f"tid{l.attr['plevel']}", )
                        idx.attr['ploop'] = l
                        new_var = Indexing(new_var, idx)
                        if i<len(to_replace[s][1])-1:
                            old_var = Indexing(old_var, idx)
                    # print(codegen.gpu.to_string([old_var, new_var]), codegen.gpu.to_string(node.compute))
                    _replace_all_ref(node.compute, old_var, new_var)
        # print(codegen.gpu.to_string(node.compute))
        ASGTraversal(replace_refs)(node)
        def reduction_procs(n, res):
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
            
            def _find_reduction_loop(s, res):
                if isinstance(s, Loop) and 'ptype' in s.attr and s.attr['ptype'] == 'reduction' and 'plevel' in s.attr and 'redu_res' not in s.attr:
                    redu_eval = None
                    for ass in s.body:
                        if type(ass) == Assignment: 
                            redu_eval = ass.lhs 
                        else:
                            continue
                    if isinstance(redu_eval, (Ndarray, Indexing)):
                        inter_res = None
                        redu_arr = get_obj(redu_eval)
                        
                        for key in to_replace:
                            if to_replace[key][0] == redu_arr:
                                inter_res = key
                                num_ext = len(to_replace[key][1])
                        if inter_res is None:
                            num_ext = len(to_replace[key][1])
                            inter_res = Ndarray(redu_eval.dtype, get_obj(redu_eval).size[:num_ext-1]+ get_obj(redu_eval).size[num_ext:])
                        
                        node.decl.append(Decl(inter_res))
                        ids = []
                        temp = redu_eval
                        while isinstance(temp, Indexing):
                            ids.insert(0, temp.idx)
                            temp = temp.dobject
                        ids = ids[:num_ext-1]+ids[num_ext:]

                        for i in ids:
                            inter_res = Indexing(inter_res, i)
                        
                        redu_loop = Loop(0, s.attr['nprocs'][-1][0], 1, [])
                        ids.insert(num_ext-1, redu_loop.iterate)
                        
                        for i in ids:
                            temp = Indexing(temp, i)
                        s.attr['redu_res'] = inter_res
                        
                        s.attr['redu_eval'] = redu_eval
                        if len(s.attr['nprocs']) > 1:
                            outerloop = s.attr['nprocs'][-2][1]
                        else:
                            outerloop = s.attr['nprocs'][-1][1]
                        
                        redu_loop.body.append(Assignment(inter_res, temp, '+'))

                        redu_loop.attr['reduction'] = True
                        redu_loop.attr['parent_loop'] = outerloop
                        res.append([redu_eval, inter_res, s, redu_loop])
                        
                        pos = []
                        if 'parent_loop' in s.attr:
                            _is_in_loopbody(s.attr['parent_loop'], outerloop.body, pos)
                            temp = outerloop.body
                            if len(pos) > 0:
                                for i in range(len(pos)-1):
                                    temp = temp[pos[i]]
                                temp.insert(pos[-1]+1, redu_loop)
                        else:
                            init_loop = Loop(0, s.attr['nprocs'][-1][0], 1, [])
                            ids[num_ext-1] = init_loop.iterate
                            temp2 = redu_arr
                            for i in ids:
                                temp2 = Indexing(temp2, i)
                            init_loop.body.append(Assignment(temp2, 0))
                            _is_in_loopbody(outerloop, node.compute, pos)
                            temp = node.compute  

                            if len(pos) > 0:
                                for i in range(len(pos)-1):
                                    temp = temp[pos[i]]
                                # temp.insert(pos[-1]+1, redu_assign)
                                temp.insert(pos[-1]+1, redu_loop)
                                temp.insert(pos[-1], init_loop)   
                            
                            def delete_init(n, res):
                                for stmt in n.compute:
                                    if type(stmt)==Assignment and _same_object(stmt.lhs.dobject, redu_arr):
                                        n.compute.remove(stmt)
                                    
                            ASGTraversal(delete_init)(node)                      

                return [True, True, True, True, True]
            
            def _replace_follow_irs(stmt, start_loop, redu_loop, old, new):
                def action(s, res):
                    global flag
                    if isinstance(s, (Loop, Assignment)):
                        if s==redu_loop:
                            return [False, False, False, False, False]
                        elif s == start_loop:
                            flag = True
                            return [False, False, False, False, False]
                        if flag:
                            _replace_all_ref(s, old, new)
                    return [True, True, True, True, True]
                IRTraversal(action)(stmt)
                
            if not 'scope' in n.attr and len(n.compute)>0:
                scope = flatten(n.compute)
                for i, s in enumerate(scope):
                    if isinstance(s, Loop):
                        res = IRTraversal(_find_reduction_loop)(s)
                        for item in res:
                            if item != []:
                                for j in range(i, len(scope)):
                                    _replace_follow_irs(scope[j], item[2], item[3], item[0], item[1])
                
            return [True, True, True, True, True]
        global flag
        flag=False
        ASGTraversal(reduction_procs)(node)

def parallelize_level(node, num_procs, level):
    loops = []
    get_loops_at_level(node.compute, level, [], loops)
    for l in loops:
        parallelize_loop(node, num_procs, l)


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
