import transform
from helpers import ASGTraversal, IRTraversal, get_obj, replace_all_ref, same_object, flatten, get_loops_at_level
from asg import *
from ir import *
import codegen
from asg2ir import gen_ir
from codegen.gpu_instruction_set import *
import copy

def _same_object(a, b):
    if isinstance(a, DObject) and isinstance(b, DObject):
        return get_obj(a).dobject_id == get_obj(b).dobject_id
    return False

def _replace_all_ref(stmt, old, new, attr=''):
    def action(s, res):
        if isinstance(s, Loop):
            if attr in s.attr and s.attr[attr]:
                return [False, False, False, False, False]
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

def add_direct_cache(node, eval):
   
    scope = flatten(node.compute)
    def get_assigns(s, res):
        if type(s) in (Assignment, Code):
            res.append(s)
        return [True, True, True, True, True]
    
    assigns = IRTraversal(get_assigns)(scope)
    

    def replace_reg(node, lhs):
        reg = Scalar(eval.dtype)
        eval.attr['cache'] = reg
        reg.attr['mem_layer'] = 'register'

        def replace_decls(n, res):
            decl = []
            for d in n.decl:
                v = d.dobject
                replace_with = None
                if _same_object(lhs, v):
                    replace_with = reg
                    if replace_with != None:
                        decl.append(Decl(replace_with))
                    else:
                        decl.append(d)
                else:
                    decl.append(d)
            n.decl = decl

        ASGTraversal(replace_decls)(node)
        
        def replace_refs(n, res):
            if not 'scope' in n.attr and len(n.compute) > 0:
                _replace_all_ref(n.compute, lhs, reg)

        ASGTraversal(replace_refs)(node)
    
    # def _find_reduction_loop(loop):
    #                 def action(s, res):
    #                     if isinstance(s, Loop):
    #                         if 'ptype' in s.attr and s.attr['ptype'] == 'reduction':
    #                             res.append(s)
    #                     return [True, True, True, True, True]

    #                 r = IRTraversal(action)(loop)
    #                 return r
    
    # for loop in scope:
    #     redu_loop = _find_reduction_loop(loop)
    #     for s in redu_loop:
    #         assign = IRTraversal(get_assigns)(s)
    #     # get lhs of last assignment
    #     lhs = assign[-1].lhs
    #     obj = get_obj(lhs)
    #     # if it is set as smem, then replace it to smem
    #     if 'mem_layer' in obj.attr and obj.attr['mem_layer'] == 'smem':
    #         replace_reg(node, lhs)
        
        
        

    
    lhs_list = []
    for s in assigns:
        if type(s) == Assignment:
            # if not _same_object(s.lhs, node.eval) and s.lhs not in lhs_list:
            if s.lhs not in lhs_list:
                lhs_list.append(s.lhs)
                
    def _get_split_loop_size(stmt, res):
        if isinstance(stmt, Loop):
            if stmt.start != 0:
                if isinstance(stmt.start, Scalar):
                    iter = stmt.start
                    if 'loop' in iter.attr:
                        loop = iter.attr['loop']
                        _get_split_loop_size(loop, res)
                        if 'nprocs' in loop.attr:
                            for jj in loop.attr['nprocs']:
                                if jj[1] == loop:
                                    res.append(jj[0])

    def replace_smem(node, lhs):
        if not same_object(lhs, node.eval):
            lhs_idx = []
            size_list = []
            temp = lhs
            while isinstance(temp, Indexing):
                lhs_idx.append(temp.idx)
                temp = temp.dobject
            flag = True

            for item in lhs_idx:
                if 'loop' in item.attr:
                    flag = False
                    loop = item.attr['loop']

                    # split loop happens
                    res = []
                    _get_split_loop_size(loop, res)
                    size_list.extend(res)

                    if 'nprocs' in loop.attr:
                        for jj in loop.attr['nprocs']:
                            if jj[1] == loop:
                                size_list.append(jj[0])
                    
                elif 'ploop' in item.attr:
                    ploop = item.attr['ploop']
                    if 'nprocs' in ploop.attr:
                        for jj in ploop.attr['nprocs']:
                            if jj[1] == ploop:
                                size_list.append(jj[0])
                        
            if flag:
                size_list = size_list[::-1]
            
            to_replace = Ndarray(arr_replace.dtype, size_list[1:])
            to_replace.attr['mem_layer'] = 'smem'
            arr_replace.attr['cache'] = to_replace
            # check if 'smem', then change to'register'
            old_eval = lhs

            def replace_decls(n, res):
                decl = []
                for d in n.decl:
                    v = d.dobject
                    replace_with = None
                    if _same_object(lhs, v):
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
                redu_idx = []
                def action(stmt, res):
                    if isinstance(stmt, Loop):
                        if stmt not in res:
                            res.append(stmt)
                        return [True, True, True, True, True]
                    elif isinstance(stmt, list|tuple):
                        for i in stmt:
                            action(i, res)
                        return [True, True, True, True, True]
                    return res
                
                def _is_val_in_loopbody(loop, old):
                    def action(s, res):
                        if _same_object(s, old):
                            res.append(True)
                        return [True, True, True, True, True]
                    return IRTraversal(action)(loop)

                if not 'scope' in n.attr and len(n.compute) > 0:
                    new_var = to_replace
                    t = IRTraversal(action)
                    res = t(n.compute)
                    
                    for item in res:
                        x = _is_val_in_loopbody(item, old_eval)
                        if x and 'plevel' in item.attr:
                            if item.attr['plevel'] == 1 or item.attr['plevel'] == 2:
                                iter = copy.deepcopy(item.iterate)
                                iter.attr['offset'] = item.start
                                idx.append(iter)
                    if new_var:
                        if idx:
                            for i in range(len(idx)):
                                if i<len(to_replace.size):
                                    new_var = Indexing(new_var, idx[i])
                        _replace_all_ref(n.compute, old_eval, new_var, 'reduction')
                        
                    new_var = to_replace
                    for item in res:
                        x = _is_val_in_loopbody(item, old_eval)
                        if x:
                            if 'reduction' in item.attr:
                                redu_idx.append(item.iterate)
                            elif 'plevel' in item.attr:
                                if item.attr['plevel'] == 1 or item.attr['plevel'] == 2:
                                    iter = copy.deepcopy(item.iterate)
                                    iter.attr['offset'] = item.start
                                    redu_idx.append(iter)
                    if new_var:
                        if redu_idx:
                            for i in range(len(redu_idx)):
                                if i<len(to_replace.size):
                                    new_var = Indexing(new_var, redu_idx[i])
                        _replace_all_ref(n.compute, old_eval, new_var)

                return [True, True, True, True, True]
            ASGTraversal(replace_refs)(node)
        
    for lhs in lhs_list:
        if lhs:
            arr_replace = get_obj(lhs)
            # if 'smem' in arr_replace.attr and arr_replace.attr['smem']:
            #     continue
            
            # for e in eval.attr['storage']:
            #     if _same_object(lhs, e):
            #         replace_smem(node, lhs)
            #         break


            # if 'mem_layer' in arr_replace.attr and arr_replace.attr['mem_layer'] == 'smem':
            #     continue

            e = eval
            while 'cache' in e.attr:
                if _same_object(lhs, e.attr['cache']) and get_obj(lhs).attr['mem_layer'] == 'global':
                    replace_smem(node, lhs)
                e = e.attr['cache']
            
            if _same_object(lhs, eval) and not _same_object(lhs, node.eval) and get_obj(lhs).attr['mem_layer'] == 'smem':
                replace_reg(node, lhs)
    

def _same_indirect_access(s1, s2):
    if isinstance(s1, Indexing) and isinstance(s2, Indexing):
        dobj1 = s1.dobject
        temp = s2
        while isinstance(temp, Indexing):
            if same_object(dobj1, temp.dobject):
                break
            temp = temp.dobject
        try:
            while isinstance(s1, (Indexing, Ndarray)) and isinstance(temp, (Indexing, Ndarray)):
                if same_object(s1.idx, temp.idx):
                    return True
                s1 = s1.idx
                temp = temp.idx
        except:
            pass
        return False

def add_indirect_cache(node, eval, C, D, *args):
    eval_list = []
    for tensor in args:
        gen_ir(tensor)
        eval_list.append(tensor.eval)
    uniq, buf, cnt = eval_list
    scope = flatten(node.compute)

    def get_indirect_access(s, res):
        if isinstance(s, (Indexing)) and _same_indirect_access(eval, s):
            res.append(s)
            return [False, False, False, False, False]
        return [True, True, True, True, True]
    inacc_list = IRTraversal(get_indirect_access)(scope)
    
    for inacc in inacc_list:
        indices = []
        temp = inacc
        while isinstance(temp, Indexing):
            if isinstance(temp.idx, Indexing):
                break
            indices.append(temp.idx)
            temp = temp.dobject
        
        cur_loop = indices[-1].attr['loop']
        par_loop = indices[-1].attr['loop'].attr['parent_loop']

        # branch for matrix and vector
        buf_idx = Indexing(buf, Literal(-1, 'int'))
        buf_idx.idx = BlockIdx()
        buf_idx = Indexing(buf_idx, Literal(-1, 'int'))
        buf_idx.idx = ThreadIdy()
        row_assign = None

        if len(eval.size) == 3:
            smem = Ndarray(eval.dtype, [2, D, D])
            smem.attr['mem_layer'] = 'smem'
            node.decl.append(Decl(smem))
            
            # data loading
            outer_loop = Loop(0, 2, 1, [])
            row_loop = Loop(ThreadIdy(), D, BlockDimy(), [])
            col_loop = Loop(ThreadIdx(), D, BlockDimx(), [])
            outer_loop.body.append(row_loop)
            row_loop.body.append(col_loop)
            global_var = Indexing(eval.dobject, Literal(-1, 'int'))
            global_var.idx = Indexing(Indexing(uniq, Literal(-1, 'int')), outer_loop.iterate)
            global_var.idx.dobject.idx = BlockIdx()
            global_var = Indexing(global_var, Expr(row_loop.iterate, indices[1].attr['loop'].attr['parent_loop'].iterate, '+'))
            global_var = Indexing(global_var, Expr(col_loop.iterate, indices[0].attr['loop'].attr['parent_loop'].iterate, '+'))

            store_smem = Indexing(smem, outer_loop.iterate)
            store_smem = Indexing(store_smem, row_loop.iterate)
            store_smem = Indexing(store_smem, col_loop.iterate)
            load = Assignment(store_smem, global_var)
            col_loop.body.append(load)

            # data access
            idx = Scalar(dtype='int')
            idx_assign = Assignment(idx, Expr(Expr(buf_idx, C, '<'), buf_idx, 'ternary', Expr(buf_idx, C, '-')))

            row_off = Scalar(dtype='int')
            row_assign = Assignment(row_off, Expr(Expr(buf_idx, C, '<'), Expr(indices[1], indices[1].attr['loop'].attr['parent_loop'].iterate, '-'), 'ternary', indices[1]))
            col_off = Scalar(dtype='int')
            col_assign = Assignment(col_off, Expr(Expr(buf_idx, C, '<'), Expr(indices[0], indices[0].attr['loop'].attr['parent_loop'].iterate, '-'), 'ternary', indices[0]))

            load_smem = Indexing(smem, idx)
            load_smem = Indexing(load_smem, row_off)
            load_smem = Indexing(load_smem, col_off)
            new_rhs = Indexing(eval.dobject, idx)
            new_rhs = Indexing(new_rhs, row_off)
            new_rhs = Indexing(new_rhs, col_off)

            res = Scalar(eval.dtype)
            res_assign = Assignment(res, Expr(Expr(buf_idx, C, '<'), load_smem, 'ternary', new_rhs))

            cur_loop.body.insert(0, res_assign)
            cur_loop.body.insert(0, col_assign)
            cur_loop.body.insert(0, row_assign)
            cur_loop.body.insert(0, idx_assign)

            node.decl.append(Decl(idx))
            node.decl.append(Decl(col_off))
            node.decl.append(Decl(res))
            node.decl.append(Decl(row_off))
            
        elif len(eval.size) == 2:
            smem = Ndarray(eval.dtype, [C, D])
            smem.attr['mem_layer'] = 'smem'
            node.decl.append(Decl(smem))

            # data loading
            cnt_access = Indexing(cnt, Literal(-1, 'int'))
            cnt_access.idx = BlockIdx()
            outer_loop = Loop(Expr(ThreadIdx(), Expr(ThreadIdy(), BlockDimx(), '*'), '+'), Expr(cnt_access, D, '*'), Expr(BlockDimx(), BlockDimy(), '*'), [])
            
            global_var = Indexing(eval.dobject, Literal(-1, 'int'))
            global_var.idx = Indexing(Indexing(uniq, Literal(-1, 'int')), Expr(outer_loop.iterate, D, '/'))
            global_var.idx.dobject.idx = BlockIdx()
            global_var = Indexing(global_var, Expr(Expr(outer_loop.iterate, D, '%'), indices[0].attr['loop'].attr['parent_loop'].iterate, '+'))

            store_smem = Indexing(smem, Expr(outer_loop.iterate, D, '/'))
            store_smem = Indexing(store_smem, Expr(outer_loop.iterate, D, '%'))
            load = Assignment(store_smem, global_var)
            outer_loop.body.append(load)

            # data access
            idx = Scalar(dtype='int')
            idx_assign = Assignment(idx, Expr(Expr(buf_idx, C, '<'), buf_idx, 'ternary', Expr(buf_idx, C, '-')))

            col_off = Scalar(dtype='int')
            col_assign = Assignment(col_off, Expr(Expr(buf_idx, C, '<'), Expr(indices[0], indices[0].attr['loop'].attr['parent_loop'].iterate, '-'), 'ternary', indices[0]))
            load_smem = Indexing(smem, idx)
            load_smem = Indexing(load_smem, col_off)
            new_rhs = Indexing(eval.dobject, idx)
            new_rhs = Indexing(new_rhs, col_off)

            res = Scalar(eval.dtype)
            res_assign = Assignment(res, Expr(Expr(buf_idx, C, '<'), load_smem, 'ternary', new_rhs))
            
            cur_loop.body.insert(0, res_assign)
            cur_loop.body.insert(0, col_assign)
            cur_loop.body.insert(0, idx_assign)
            
            node.decl.append(Decl(idx))
            node.decl.append(Decl(col_off))
            node.decl.append(Decl(res))


        outer_loop.attr['load'] = True
        par_loop.body.insert(0, SyncThreads())
        par_loop.body.insert(0, outer_loop)
        outer_loop.attr['parent_loop'] = par_loop
    
        # replace all refs
        replace_all_ref(node.compute, inacc, res)
        
        
