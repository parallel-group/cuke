import transform
from helpers import ASGTraversal, IRTraversal, get_obj, replace_all_ref, same_object, flatten, get_loops_at_level
from asg import *
from ir import *
import codegen
from asg2ir import gen_ir
import copy

def _same_object(a, b):
    if isinstance(a, DObject) and isinstance(b, DObject):
        return get_obj(a).dobject_id == get_obj(b).dobject_id
    return False

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
                if _same_object(s.val, old):
                    s.val = new
            case 'Code':
                if _same_object(s.output[1], old):
                    s.output = (s.output[0], new)
                for k in s.inputs:
                    if s.inputs[k] == old:
                        s.inputs[k] = new
        return [True, True, True, True, True]

    t = IRTraversal(action)
    res = t(stmt)

def apply_smem(node, eval, attr=''):
    C=16
    D=64
    
    parent_size = 0
    for i in node.ref_by:
        if i.op_type == 'apply':
            parent_size = 1

    
    scope = flatten(node.compute)

    def get_assigns(s, res):
        if type(s) in (Assignment, Code):
            res.append(s)
        return [True, True, True, True, True]
    # print(codegen.gpu.to_string(scope))
    assigns = IRTraversal(get_assigns)(scope)
    print(assigns, codegen.gpu.to_string(assigns))
    lhs = None
    lhs_list = []
    for s in assigns:
        if type(s) == Assignment:
            # lhs = s.lhs 
            if not _same_object(s.lhs, eval) and s.lhs not in lhs_list:
                lhs_list.append(s.lhs)
    
    def replace_smem(node, lhs):
        if not same_object(lhs, eval):
            to_replace = Ndarray(arr_replace.dtype, arr_replace.size[1:])
            to_replace.attr['smem'] = True
            old_eval = lhs
            
            def replace_decls(n, res):
                for nn in n.ref_by:
                    replace_decls(nn, res)
                decl = []
                for d in n.decl:
                    v = d.dobject
                    replace_with = None

                    if _same_object(lhs, v) or (n.eval != eval and _same_object(n.eval, v)):
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
                        _replace_all_ref(n.compute, old_eval, new_var)

                return [True, True, True, True, True]
            ASGTraversal(replace_refs)(node)

    for lhs in lhs_list:
        if lhs:
            arr_replace = get_obj(lhs)
            
            if 'smem' in arr_replace.attr and arr_replace.attr['smem']:
                continue
            print(codegen.gpu.to_string(lhs), codegen.gpu.to_string(eval), not same_object(lhs, eval), arr_replace.size, arr_replace.attr)
            
            if (attr != '' and 'mem_opt' in arr_replace.attr) or (attr == '' and 'mem_opt' not in arr_replace.attr):
                if attr != '' and arr_replace.attr['mem_opt'] == attr:
                    # replace mem to smem
                    replace_smem(node, lhs)
                elif attr == '':
                    replace_smem(node,lhs)

            



def gather_smem(node, C, D):
    scope = flatten(node.compute)

    def get_assigns(s, res):
        if type(s) in (Assignment, Code):
            res.append(s)
        return [True, True, True, True, True]
    
    assigns = IRTraversal(get_assigns)(scope)

    def is_reused(s):
        def action(stmt, res):
            # if isinstance(stmt, IR):
            #     print('--->', stmt, codegen.gpu.to_string(stmt), stmt.attr)
            if isinstance(stmt, IR) and 'reuse' in stmt.attr and stmt.attr['reuse']:
                
                res.append(stmt)
            return [True, True, True, True, True]
        x = IRTraversal(action)(s)
        return x

    x = is_reused(scope)
    
    if x != []:
        print(x, codegen.gpu.to_string(x))
        # print(assigns, codegen.gpu.to_string(assigns))
        for i in assigns:
            print(codegen.gpu.to_string(i), i.attr)
            if 'parent_loop' in i.attr:
                print(codegen.gpu.to_string(i.attr['parent_loop']))
