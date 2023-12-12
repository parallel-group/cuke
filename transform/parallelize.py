import transform
from helpers import ASGTraversal, IRTraversal, get_obj, replace_all_ref, same_object, flatten
from asg import *
from ir import *
import codegen
from asg2ir import gen_ir




def _get_ref_idx(stmt, v):
    def _get_scalar_idx(idx):
        assert type(idx) == Indexing
        def action(s, res):
            if type(s) == Scalar:
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





class parallelizer:

    def __init__(self, num_procs=[16]):
        self.num_procs = num_procs

    def __call__(self, node):

        def find_all_vars(n, res):
            res.extend([d.dobject for d in n.decl])

        all_vars = ASGTraversal(find_all_vars)(node)

        def ir_set_loop_level(stmt, num_procs):
            def action(s, res):
                if type(s) in (list, tuple):
                    return [True]
                if isinstance(s, Loop):
                    if not 'plevel' in s.attr and not 'nprocs' in s.attr:
                        s.attr['plevel'] = 0
                        s.attr['nprocs'] = [(num_procs[0], s)]
                    for ss in flatten(s.body):
                        if isinstance(ss, Loop):
                            if 'plevel' in s.attr and s.attr['plevel'] + 1 < len(num_procs):
                                ss.attr['plevel'] = s.attr['plevel'] + 1
                                ss.attr['nprocs'] = s.attr['nprocs'] + [(num_procs[ss.attr['plevel']], ss)]
                            else:
                                ss.attr['nprocs'] = s.attr['nprocs']
                    if type(s) == FilterLoop:
                        for ss in flatten(s.cond_body):
                            if isinstance(ss, Loop):
                                if 'plevel' in s.attr and s.attr['plevel'] + 1 < len(num_procs):
                                    ss.attr['plevel'] = s.attr['plevel'] + 1
                                    ss.attr['nprocs'] = s.attr['nprocs'] + [(num_procs[ss.attr['plevel']], ss)]
                                else:
                                    ss.attr['nprocs'] = s.attr['nprocs']
                    return [False, False, False, True, True]

                return [False, False, False, False, False]

            IRTraversal(action)(stmt)

        def ir_find_data_races(stmt, to_replace):
            def action(s, res):
                if isinstance(s, Loop):
                    if 'nprocs' in s.attr:
                        assigns = [ss for ss in flatten(s.body) if type(ss) in (Assignment, Code)]
                        if type(s) == FilterLoop:
                            assigns.extend([ss for ss in flatten(s.cond_body) if type(ss) in (Assignment, Code)])
                        for ss in assigns:
                            for v in all_vars:
                                if v not in to_replace:
                                    lhs = ss.lhs if type(ss) == Assignment else ss.output
                                    idx = _get_ref_idx(lhs, v)
                                    if idx != None:
                                        ext_size = []
                                        loop_info = []
                                        for l in s.attr['nprocs']:
                                            indexed = False
                                            for ii in idx:
                                                if ('output_axis' in l[1].attr and l[1].attr['output_axis'] ==
                                                    ii.attr['loop'].attr['output_axis']) or same_object(ii, l[
                                                    1].iterate):
                                                    indexed = True
                                                    break
                                            if not indexed:
                                                ext_size.append(l[0])
                                                loop_info.append(l[1])
                                        if len(ext_size) > 0:
                                            to_replace[v] = (Ndarray(v.dtype, ext_size + v.size), loop_info)

                return [True, True, True, True, True]

            IRTraversal(action)(stmt)


        def find_shared_vars(n, res):
            if len(res) == 0:
                res.append({})
            if not 'scope' in n.attr and len(n.compute) != 0:
                ir_set_loop_level(n.compute, self.num_procs)
                ir_find_data_races(n.compute, res[0])

        to_replace = ASGTraversal(find_shared_vars)(node)[0]

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
                else:
                    decl.append(d)
            n.decl = decl

        ASGTraversal(replace_decls)(node)

        def replace_refs(n, res):
            if len(n.compute) > 0:
                for s in to_replace:
                    new_var = to_replace[s][0]
                    for l in to_replace[s][1]:
                        new_var = Indexing(new_var, Literal(f"tid{l.attr['plevel']}", 'int'))
                    replace_all_ref(n.compute, s, new_var)

        ASGTraversal(replace_refs)(node)
        return node


def test1():
    A = Tensor((10, 20))
    B = Tensor((10, 20))
    C = Tensor((10, 20))
    D = Tensor((10, 20))

    A = setval(A, 1)
    t1 = A + B
    t2 = (C - D).abs()
    res1 = t1 + t2
    transform.passes.append(parallelizer())
    code = codegen.cpu.print_cpp(gen_ir(res1))
    print(code)

def test2():
    A = Tensor((10, 20))
    B = Tensor((20, 30))
    C = Tensor((10, 30))
    D = Tensor((10, 30))
    t1 = (A @ B).abs()
    t2 = (C - D).abs()
    res1 = t1 + t2
    transform.passes.append(parallelizer())
    code = codegen.cpu.print_cpp(gen_ir(res1))
    print(code)

def test3():
    A = Tensor((10, 20))
    B = Tensor((20, 30))
    C = Tensor((10, 20))
    D = Tensor((20, 30))
    t1 = (A @ B).abs()
    t2 = (C @ D).round()
    res1 = t1 + t2
    transform.passes.append(parallelizer())
    print(codegen.cpu.print_cpp(gen_ir(res1)))



if __name__ == "__main__":
    # test1()
    # test2()
    test3()