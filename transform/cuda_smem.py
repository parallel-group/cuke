import transform
from helpers import ASGTraversal, IRTraversal, get_obj, replace_all_ref, same_object, flatten, get_loops_at_level
from asg import *
from ir import *
import codegen
from asg2ir import gen_ir
import copy

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

def apply_smem(node, eval, C, D):
    
    parent_size = 0
    for i in node.ref_by:
        if i.op_type == 'apply':
            parent_size = 1

    
    scope = flatten(node.compute)

    def get_assigns(s, res):
        if type(s) in (Assignment, Code):
            res.append(s)
        return [True, True, True, True, True]
    
    assigns = IRTraversal(get_assigns)(scope)
    
    lhs = None
    for s in assigns:
        if type(s) == Assignment:
            lhs = s.lhs 
            break
    
    if lhs:
        if node.eval != eval and not same_object(lhs, eval):
            to_replace = None
            old_eval = None
            
            if len(node.eval.size) + parent_size == 1:
                to_replace = Ndarray(node.eval.dtype, [C])
                to_replace.attr['smem'] = True
                old_eval = lhs
                
            elif len(node.eval.size) + parent_size == 2:
                
                to_replace = Ndarray(node.eval.dtype, [C, D])
                to_replace.attr['smem'] = True
                old_eval = lhs
                
            def replace_decls(n, res):
                for nn in n.ref_by:
                    replace_decls(nn, res)
                decl = []
                for d in n.decl:
                    v = d.dobject
                    replace_with = None
                    if same_object(lhs, v) or (n.eval != eval and same_object(n.eval, v)):
                        replace_with = to_replace
                        if replace_with != None:
                            decl.append(Decl(replace_with))
                        else:
                            decl.append(d)
                    else:
                        decl.append(d)
                n.decl = decl

            ASGTraversal(replace_decls)(node)
            
            def replace_refs(n, res):
                idx = []
                def action(stmt, res):
                    if isinstance(stmt, Loop):
                        if stmt.iterate not in res:
                            res.append(stmt.iterate)
                            res.append(stmt)
                        return [True, True, True, True, True]
                    elif isinstance(stmt, list|tuple):
                        for i in stmt:
                            action(i, res)
                        return [True, True, True, True, True]
                    return res
                for nn in n.ref_by:
                    replace_refs(nn, res)
                if len(n.compute) > 0:
                    new_var = to_replace
                    t = IRTraversal(action)
                    res = t(n.compute)
                    for item in res:
                        if 'plevel' in item.attr:
                            if item.attr['plevel'] == 1 or item.attr['plevel'] == 2:
                                iter = copy.deepcopy(item.iterate)
                                iter.attr['offset'] = item.start
                                idx.append(iter)
                    if new_var:
                        if idx:
                            for i in range(len(idx)):
                                if i<len(to_replace.size):
                                    new_var = Indexing(new_var, idx[i])
                        replace_all_ref(n.compute, old_eval, new_var)

                return [True, True, True, True, True]
            ASGTraversal(replace_refs)(node)



def gather_smem(node, C, D):
    scope = flatten(node.compute)

    def get_assigns(s, res):
        if type(s) in (Assignment, Code):
            res.append(s)
        return [True, True, True, True, True]
    
    assigns = IRTraversal(get_assigns)(scope)
    print(codegen.gpu.to_string(scope))

    def is_reused(s):
        def action(stmt, res):
            # if isinstance(stmt, IR):
            #     print('--->', stmt, codegen.gpu.to_string(stmt), stmt.attr)
            if isinstance(stmt, IR) and 'reuse' in stmt.attr and stmt.attr['reuse']:
                
                res.append(stmt)
                # print('<<<<', stmt, codegen.gpu.to_string(stmt), stmt.attr, res)
            return [True, True, True, True, True]
        x = IRTraversal(action)(s)
        return x

    x = is_reused(scope)
    print(x, codegen.gpu.to_string(x), codegen.gpu.to_string(node.eval))
