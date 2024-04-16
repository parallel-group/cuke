from __future__ import annotations
import copy
from . import ir
from . import asg
from . import helpers


def num_unbind(index):
    if type(index) == ir.Indexing:
        return num_unbind(index.dobject) + num_unbind(index.idx)
    elif type(index) == ir.Literal and index.val == -1:
        return 1
    else:
        return 0

def bind(object: ir.Indexing | ir.Ndarray | ir.Slice, subscripts: list | tuple, attrs = None):
    new_index = copy.deepcopy(object)
    # if attrs == None:
    #     attrs = [{} for _ in range(len(subscripts))]
    j = 0
    if type(new_index) == ir.Indexing:
        indices = [new_index]
        while type(indices[-1].dobject) == ir.Indexing:
            indices.append(indices[-1].dobject)
        indices.reverse()
        i = 0
        while i < len(indices) and j < len(subscripts):
            index = indices[i]
            i += 1
            while type(index.idx) == ir.Indexing:
                index = index.idx
            assert type(index.idx) in (ir.Scalar, ir.Literal, ir.Expr)
            if type(index.idx) == ir.Scalar or type(index.idx) == ir.Expr or (type(index.idx) == ir.Literal and index.idx.val != -1):
                continue
            idx = subscripts[j]
            if type(idx) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Expr):
                index.idx = idx
            elif type(idx) in (ir.Ndarray, ir.Slice):
                index.idx = ir.Indexing(idx, ir.Literal(-1, 'int'))
            else:
                raise TypeError('idx type error when binding')
            # index.attr.update(attrs[j])
            index.refresh_size()
            j += 1

    while j < len(subscripts):
        idx = subscripts[j]
        if type(idx) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Expr):
            new_index = ir.Indexing(new_index, idx)
        elif type(idx) in (ir.Ndarray, ir.Slice):
            new_index = ir.Indexing(new_index, ir.Indexing(idx, ir.Literal(-1, 'int')))
        else:
            raise TypeError('incorrect idx type!')
        # new_index.attr.update(attrs[j])
        j += 1
    if type(new_index) == ir.Indexing:
        new_index.refresh_size()
    return new_index



def get_slice(index: (ir.Indexing, ir.Ndarray, ir.Slice)):
    if type(index) == ir.Indexing:
        x = get_slice(index.dobject)
        if x != None:
            return x
        else:
            y = get_slice(index.idx)
            if y != None:
                return y
            else:
                if type(index.dobject) == ir.Slice and type(index.idx) == ir.Literal and index.idx.val == -1:
                    return index.dobject
    return None


def replace_output(stmt, old, new):
    if type(stmt) == list or type(stmt) == tuple:
        for l in stmt:
            replace_output(l, old, new)
    elif type(stmt) == ir.Loop:
        replace_output(stmt.body, old, new)
    elif type(stmt) == ir.Assignment:
        if stmt.lhs == old:
            stmt.lhs = new
        else:
            replace_output(stmt.lhs, old, new)
    elif type(stmt) == ir.Indexing:
        if stmt.dobject == old:
            stmt.dobject = new
        else:
            replace_output(stmt.dobject, old, new)


def list_product(l):
    res = ir.Literal(1, dtype='int')
    for x in l:
        if not type(x) in (tuple, list):
            res = ir.Expr(res, x, '*')
        else:
            res = ir.Expr(res, list_product(x), '*')
    return res

def get_subdims(x, dims, sizes):
    res = dict()
    y = x
    for i in range(len(sizes)):
        d = dims[len(sizes) - 1 - i]
        s = sizes[len(sizes) - 1 - i]
        if not type(d) in (tuple, list):
            if d != -1:
                if d not in res:
                    res[d] = [(ir.Expr(y, s, '%') if i < len(sizes) - 1 else y, s)]
                else:
                    res[d].insert(0, (ir.Expr(y, s, '%') if i < len(sizes) - 1 else y, s))
        else:
            ts = list_product(s)
            sd = get_subdims(ir.Expr(y, ts, '%') if i < len(sizes) - 1 else y, d, s)
            for k in sd:
                if k not in res:
                    res[k] = sd[k]
                else:
                    res[k][0:0] = sd[k]
        y = ir.Expr(y, s, '/')

    return res

def resolve_view(node, subscripts):
    if isinstance(node, asg.Tensor):
        if 'dim_map' in node.attr and 'size_map' in node.attr:
            dim_map = node.attr['dim_map']
            size_map = node.attr['size_map']
            assert len(dim_map) == len(subscripts)
            assert helpers.list_same_size(dim_map, size_map)

            orig_subscripts = dict()
            for i in range(len(dim_map)):
                d = dim_map[i]
                if type(d) in (list, tuple):
                    t = get_subdims(subscripts[i], d, size_map[i])
                    for k in t:
                        if k != -1:
                            if k in orig_subscripts:
                                orig_subscripts[k].extend(t[k])
                            else:
                                orig_subscripts[k] = t[k]
                elif d != -1: # remove fake axes
                    if d in orig_subscripts:
                        orig_subscripts[d].append((subscripts[i], size_map[i]))
                    else:
                        orig_subscripts[d] = [(subscripts[i], size_map[i])]

            res_subscripts = []
            for k in sorted(orig_subscripts.keys()):
                sg = [tmp[0] for tmp in orig_subscripts[k]]
                ds = [tmp[1] for tmp in orig_subscripts[k]]
                orig_s = sg[0]

                for j in range(1, len(sg)):
                    if type(orig_s) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Ndarray, ir.Expr):
                        if type(sg[j]) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Ndarray, ir.Expr):
                            orig_s = ir.Expr(ir.Expr(orig_s, ds[j], '*'), sg[j], '+')
                        elif type(sg[j]) == ir.Slice:
                            start = ir.Expr(ir.Expr(orig_s, ds[j], '*'), sg[j].start, '+')
                            stop = ir.Expr(ir.Expr(orig_s, ds[j], '*'), sg[j].stop, '+')
                            orig_s = ir.Slice(start, stop, sg[j].step)
                        else:
                            raise TypeError('idx type error')
                    elif type(orig_s) == ir.Slice:
                        if type(sg[j]) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Ndarray, ir.Expr):
                            start = ir.Expr(ir.Expr(orig_s.start, ds[j], '*'), sg[j], '+')
                            stop = ir.Expr(
                                ir.Expr(ir.Expr(ir.Expr(orig_s.stop, 1, '-'), ds[j], '*'), sg[j], '+'), 1,
                                '+')
                            step = ir.Expr(orig_s.step, ds[j], '*')
                            orig_s = ir.Slice(start, stop, step)
                        elif type(sg[j]) == ir.Slice:
                            start = ir.Expr(ir.Expr(orig_s.start, ds[j], '*'), sg[j].start, '+')
                            stop = ir.Expr(ir.Expr(ir.Expr(orig_s.stop, 1, '-'), ds[j], '*'), sg[j].stop, '+')
                            assert (orig_s.step == 1 or orig_s.step.val == 1) and (sg[j].step == 1 or sg[
                                j].step.val == 1), 'view does not support non-continuous slice of slice'
                            orig_s = ir.Slice(start, stop, 1)
                        else:
                            raise TypeError('idx type error')
                    else:
                        raise TypeError('orig type error')
                res_subscripts.append(orig_s)
            return res_subscripts
    return subscripts


def gen_ir(node):
    assert isinstance(node, asg.ASTNode)
    if node.eval or len(node.decl) > 0 or (type(node) == asg.TensorOp and len(node.compute) > 0):
        return node
    if type(node) == asg.Const:
        if node.dtype != 'slice':
            assert type(node.val) == int or type(node.val) == float
            node.eval = ir.Literal(node.val, node.dtype)
        else:
            gen_ir(node.val.start)
            gen_ir(node.val.stop)
            gen_ir(node.val.step)
            node.eval = ir.Slice(node.val.start.eval, node.val.stop.eval, node.val.step.eval)


    elif type(node) == asg.Var or (type(node) == asg.Tensor and len(node._size()) == 0):
        node.eval = ir.Scalar(node.dtype, node.name)
        for key in node.attr:
            node.eval.attr[key] = node.attr[key]
        node.decl = [ir.Decl(node.eval)]

    elif type(node) == asg.Tensor and len(node._size()) > 0:
        # convert AST sizes to IR sizes
        size = helpers.get_ir_of_size(node._size())
        node.eval = ir.Ndarray(node.dtype, size, node.name)
        for key in node.attr:
            node.eval.attr[key] = node.attr[key]
        node.decl = [ir.Decl(node.eval)]

    elif type(node) == asg.TensorOp:
        if node.op_type in asg.arith_op or node.op_type in asg.cmp_op:
            # arith_op and cmp_op are binary operations, we generate the two operands first
            gen_ir(node.operators[0])
            gen_ir(node.operators[1])
            assert isinstance(node.operators[0], asg.Tensor) and isinstance(node.operators[1], asg.Tensor)

            if node.op_type in asg.arith_op:
                op = asg.arith_op[node.op_type]
            else:
                op = node.op_type

            if len(node._size()) > 0:  # if output has >=1 dimensions, it should be stored in an Ndarray
                size = helpers.get_ir_of_size(node._size())
                node.eval = ir.Ndarray(node.dtype, size)

                node.decl = [ir.Decl(node.eval)]


                left_levels = len(node.operators[0]._size())
                right_levels = len(node.operators[1]._size())
                max_levels = max(left_levels, right_levels)
                assert max_levels == len(size)

                lhs = node.operators[0].eval
                rhs = node.operators[1].eval
                res = node.eval
                compute = node.compute

                lhs_subscripts = []
                rhs_subscripts = []
                res_subscripts = []

                if compute == []:
                    par_loop = None
                else:
                    par_loop = compute[0]
                for level in range(max_levels):

                    # handle out of bound slicing
                    # left_slice = get_slice(lhs)
                    # right_slice = get_slice(rhs)
                    # left_attr = {}
                    # if left_slice != None and type(left_slice.start) == ir.Literal:
                    #     if left_slice.start.val < 0:
                    #         left_ofs = -left_slice.start.val
                    #         left_attr['slice_ofs'] = left_ofs
                    #     else:
                    #         left_ofs = 0
                    # else:
                    #     left_ofs = 0
                    # right_attr = {}
                    # if right_slice != None and type(right_slice.start) == ir.Literal:
                    #     if right_slice.start.val < 0:
                    #         right_ofs = -right_slice.start.val
                    #         right_attr['slice_ofs'] = right_ofs
                    #     else:
                    #         right_ofs = 0
                    # else:
                    #     right_ofs = 0

                    pre_loop = ir.Loop(0, size[level], 1, [])
                    # loop_ofs = max(left_ofs, right_ofs)
                    # if loop_ofs > 0:
                    #     pre_loop.attr['loop_ofs'] = loop_ofs

                    if level < left_levels:
                        lhs_subscripts.append(pre_loop.iterate)
                        node.input_orders[0].append((level, pre_loop))
                    if level < right_levels:
                        rhs_subscripts.append(pre_loop.iterate)
                        node.input_orders[1].append((level, pre_loop))
                    res_subscripts.append(pre_loop.iterate)
                    node.output_order.append((level, pre_loop))
                    pre_loop.attr['output_axis'] = level
                    if par_loop:
                        pre_loop.attr['parent_loop'] = par_loop
                    else:
                        par_loop = pre_loop
                    compute.append(pre_loop)
                    compute = pre_loop.body

                lhs = bind(lhs, resolve_view(node.operators[0], lhs_subscripts))
                rhs = bind(rhs, resolve_view(node.operators[1], rhs_subscripts))
                res = bind(res, res_subscripts)
                compute.append(ir.Assignment(res, ir.Expr(lhs, rhs, op)))
            else:
                node.eval = ir.Expr(node.operators[0].eval, node.operators[1].eval, op)

        elif node.op_type in asg.math_op:
            gen_ir(node.operators[0])

            if len(node._size()) > 0:
                size = helpers.get_ir_of_size(node._size())
                node.eval = ir.Ndarray(node.dtype, size)


                node.decl = [ir.Decl(node.eval)]

                res = node.eval
                val = node.operators[0].eval
                levels = len(size)
                compute = node.compute

                subscripts = []
                for level in range(levels):
                    # sl = get_slice(val)
                    # attr = {}
                    # if sl != None and type(sl.start) == ir.Literal:
                    #     if sl.start.val < 0:
                    #         ofs = -sl.start.val
                    #         attr['slice_ofs'] = ofs
                    #     else:
                    #         ofs = 0
                    # else:
                    #     ofs = 0

                    pre_loop = ir.Loop(0, size[level], 1, [])
                    # if ofs > 0:
                    #     pre_loop.attr['loop_ofs'] = ofs

                    subscripts.append(pre_loop.iterate)
                    node.input_orders[0].append((level, pre_loop))
                    node.output_order.append((level, pre_loop))
                    pre_loop.attr['output_axis'] = level
                    compute.append(pre_loop)
                    compute = pre_loop.body

                val = bind(val, resolve_view(node.operators[0], subscripts))
                res = bind(res, subscripts)

                compute.append(ir.Assignment(res, ir.Math(val, node.op_type)))
            else:
                node.eval = ir.Math(node.operators[0].eval, node.op_type)

        elif node.op_type == 'setval':
            if type(node.operators[0]) == asg.Tensor:
                node.operators[0].attr['is_arg'] = False

            gen_ir(node.operators[0])
            gen_ir(node.operators[1])

            node.eval = node.operators[0].eval

            if helpers.is_scalar(node.operators[1]):
                val = node.operators[1].eval

                if len(node.ref_size) > 0:
                    size = helpers.get_ir_of_size(node.ref_size)
                    pre_loop = ir.Loop(0, size[0], 1, [])
                    node.compute = [pre_loop]
                    res = bind(node.eval, [pre_loop.iterate])
                    for i in range(1, len(size)):
                        loop = ir.Loop(0, size[i], 1, [])
                        pre_loop.body.append(loop)
                        pre_loop = loop
                        res = bind(res, [pre_loop.iterate])

                    assign = ir.Assignment(res, val)
                    pre_loop.body.append(assign)
                else:
                    node.compute = [ir.Assignment(node.eval, val)]

                l = node.compute[0]
                for i in range(len(node.eval.size)):
                    node.output_order.append((i, l))
                    l.attr['output_axis'] = i
                    l = l.body[0]
            else:
                node.operators[1].decl = [d for d in node.operators[1].decl if d.dobject != node.operators[1].eval]
                # find all defs and replace them with new node eval
                for dfs in helpers.ir_find_defs(node.operators[1].compute, node.operators[1].eval):
                    if isinstance(dfs.lhs, ir.Indexing):
                        temp = dfs.lhs
                        idx_list = []
                        while isinstance(temp, ir.Indexing):
                            idx_list.append(temp.idx)
                            temp = temp.dobject
                        idx_list.reverse()
                        res = bind(node.eval, idx_list)
                        helpers.replace_all_ref(node.operators[1].compute, dfs.lhs, res)
                    else:
                        replace_output(node.operators[1].compute, node.operators[1].eval, node.eval)
                node.operators[1].eval = node.eval
                node.output_order = node.operators[1].output_order

        elif node.op_type == 'einsum':
            gen_ir(node.operators[0])
            if node.operators[1] != None:
                gen_ir(node.operators[1])
            node.input_orders[0] = []
            node.input_orders[1] = []

            exp = node.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            all_indices = ''.join(sorted(set(input1 + input2)))
            all_loops = []
            mapping = {}

            reduce_begins = len(output)

            for i in output:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if (pos1 >= 0 and pos2 < 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                elif (pos1 < 0 and pos2 >= 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[1].eval.ref_size(pos2), 1, [])
                    all_loops.append(l)
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                if i in output:
                    continue
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if (pos1 >= 0 and pos2 < 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                elif (pos1 < 0 and pos2 >= 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[1].eval.ref_size(pos2), 1, [])
                    all_loops.append(l)
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if pos1 >= 0 and pos2 >= 0:
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if pos1 < 0 and pos2 < 0:
                    raise IndexError('index not found!')

            for i in range(reduce_begins, len(all_loops)):
                all_loops[i].attr['ptype'] = 'reduction'

            op1 = node.operators[0].eval
            op1_subscripts = []
            for i in input1:
                op1_subscripts.append(all_loops[mapping[i]].iterate)
            op1 = bind(op1, resolve_view(node.operators[0], op1_subscripts))

            if node.operators[1] != None:
                op2 = node.operators[1].eval
                op2_subscripts = []
                for i in input2:
                    op2_subscripts.append(all_loops[mapping[i]].iterate)
                op2 = bind(op2, resolve_view(node.operators[1], op2_subscripts))
            else:
                op2 = None

            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = ir.Ndarray(node.dtype, size)
            else:
                node.eval = ir.Scalar(node.dtype)
            node.decl = [ir.Decl(node.eval)]
            res = node.eval
            for i in output:
                res = bind(res, [all_loops[mapping[i]].iterate])

            if op2 != None:
                expr = ir.Expr(op1, op2, '*')
            else:
                expr = op1
            if reduce_begins == len(all_loops):
                body = ir.Assignment(res, expr)
            else:
                body = ir.Assignment(res, expr, '+')
            init = ir.Assignment(res, 0)
            if reduce_begins == 0:
                node.compute.append(init)
            pre_loop = all_loops[0]
            node.compute.append(pre_loop)
            for i in range(1, len(all_loops)):
                if reduce_begins == i:
                    init.attr['parent_loop'] = pre_loop
                    pre_loop.body.append(init)
                loop = all_loops[i]
                loop.attr['parent_loop'] = pre_loop
                pre_loop.body.append(loop)
                pre_loop = loop
            body.attr['parent_loop'] = pre_loop
            pre_loop.body.append(body)

            l = node.compute[0]
            for i in range(len(node.eval.size)):
                node.output_order.append((i, l))
                l.attr['output_axis'] = i
                l = l.body[0]

        elif node.op_type == 'view':
            gen_ir(node.operators[0])
            node.eval = node.operators[0].eval
            dim_map = []
            size_map = []
            if 'size_map' in node.operators[0].attr:
                ref_size1 = node.operators[0].attr['size_map']
            else:
                ref_size1 = helpers.get_ir_of_size(node.operators[0].ref_size)
            ref_size2 = helpers.get_ir_of_size(node.ref_size)
            for i in range(len(node.operators[1])):
                s = node.operators[1][i]
                if type(s) in (list, tuple):
                    d = []
                    si = []
                    for ss in s:
                        assert type(ss) == asg.Const
                        val = ss.val
                        si.append(ref_size1[val])
                        if 'dim_map' in node.operators[0].attr:
                            val = node.operators[0].attr['dim_map'][val]
                        d.append(val)
                    dim_map.append(d)
                    size_map.append(si)
                else:
                    assert type(s) == asg.Const
                    val = s.val
                    if 'dim_map' in node.operators[0].attr:
                        val = node.operators[0].attr['dim_map'][val]
                    dim_map.append(val)
                    size_map.append(ref_size2[i])
            node.attr['dim_map'] = dim_map
            node.attr['size_map'] = size_map

        elif node.op_type == 'index':
            gen_ir(node.operators[0])
            subscripts = []
            for op in node.operators[1:]:
                gen_ir(op)
                subscripts.append(op.eval)
            for i in range(len(subscripts), len(node.operators[0].ref_size)):
                op = asg.Const(slice(0, node.operators[0].ref_size[i], 1), 'slice')
                gen_ir(op)
                subscripts.append(op.eval)

            real_subscripts = resolve_view(node.operators[0], subscripts)
            assert len(real_subscripts) == len(node.operators[0].eval.size)
            node.eval = bind(node.operators[0].eval, real_subscripts)

            for key in node.operators[0].eval.attr:
                node.eval.attr[key] = node.operators[0].eval.attr[key]

            if 'dim_map' in node.operators[0].attr:
                dim_map = [node.operators[0].attr['dim_map'][i] for i in range(len(subscripts)) if type(subscripts[i]) == ir.Slice]
                size_map = [node.operators[0].attr['size_map'][i] for i in range(len(subscripts)) if type(subscripts[i]) == ir.Slice]
                tmp = list(range(len(real_subscripts)))
                j = 0
                for i in range(len(real_subscripts)):
                    if type(real_subscripts[i]) == ir.Slice:
                        tmp[i] = j
                        j += 1
                    else:
                        tmp[i] = None
                node.attr['dim_map'] = []
                node.attr['size_map'] = []
                for i in range(len(dim_map)):
                    d = dim_map[i]
                    if d == -1:
                        node.attr['dim_map'].append(-1)
                        node.attr['size_map'].append(size_map[i])
                    else:
                        if tmp[d] is not None:
                            node.attr['dim_map'].append(tmp[d])
                            node.attr['size_map'].append(size_map[i])


        elif node.op_type == 'apply':

            # operators: func, data (node.nparams), axis (node.nparams), cond, items (node.nparams), ret, counter

            # evaluate data, axis, cond
            for i in range(1, 2 + 2 * node.nparams):
                if node.operators[i] != None:
                    gen_ir(node.operators[i])

            primary_axis = node.operators[1 + node.nparams].eval.val
            sizes =  helpers.get_ir_of_size(node.operators[1].ref_size)

            # this is the loop that iterates over the axis of the primary (first) tensor input
            cond = node.operators[1 + 2 * node.nparams]
            if cond == None:
                outer_loop = ir.Loop(0, sizes[primary_axis], 1, [])
            else:
                outer_loop = ir.FilterLoop(0, sizes[primary_axis], 1,
                                        cond.eval, [], [])
                # gen ir for the counter
                gen_ir(node.operators[-1])


            nn = []
            for i in range(node.nparams):
                data = node.operators[1 + i]
                axis = node.operators[1 + node.nparams + i].eval.val

                # number of unbind axes in the eval of input data
                n = num_unbind(data.eval)
                nn.append(n)

                subscripts = []

                for j in range(axis):
                    op = asg.Const(slice(0, data.ref_size[j], 1), 'slice')
                    gen_ir(op)
                    subscripts.append(op.eval)

                subscripts.append(outer_loop.iterate)

                for j in range(axis+1, len(data.ref_size)):
                    op = asg.Const(slice(0, data.ref_size[j], 1), 'slice')
                    gen_ir(op)
                    subscripts.append(op.eval)

                real_subscripts = resolve_view(data, subscripts)

                item = node.operators[2 + 2 * node.nparams + i]
                item.eval = bind(data.eval, real_subscripts)

                if 'dim_map' in data.attr and 'size_map' in data.attr:
                    dim_map = [data.attr['dim_map'][ii] for ii in range(len(subscripts)) if
                               type(subscripts[ii]) == ir.Slice]
                    size_map = [data.attr['size_map'][ii] for ii in range(len(subscripts)) if
                                type(subscripts[ii]) == ir.Slice]
                    item = node.operators[2 + 2 * node.nparams + i]
                    tmp = list(range(len(real_subscripts)))
                    j = 0
                    for k in range(len(real_subscripts)):
                        if type(real_subscripts[k]) == ir.Slice:
                            tmp[k] = j
                            j += 1
                        else:
                            tmp[k] = None
                    item.attr['dim_map'] = []
                    item.attr['size_map'] = []
                    for k in range(len(dim_map)):
                        d = dim_map[k]
                        if d == -1:
                            item.attr['dim_map'].append(-1)
                            item.attr['size_map'].append(size_map[k])
                        else:
                            if tmp[d] is not None:
                                item.attr['dim_map'].append(tmp[d])
                                item.attr['size_map'].append(size_map[k])

            # since input items of func has been generated and indexed, we can generate the IR of the func
            ret = node.operators[-2]
            gen_ir(ret)

            # get the input orders
            for i in range(min(len(ret.input_orders), node.nparams)):
                n = nn[i]
                l = node.input_orders[1 + i]
                axis = node.operators[1 + node.nparams + i].eval.val
                # TODO: (Yihua) I don't remember how this is implemented, Yihua can you add some comments to explain it?
                # TODO: input orders may need update for views
                if axis >= n:
                    for j in range(axis):
                        l.append((len(l), ret.input_orders[i][j][1]))
                    l.append((len(l), outer_loop))
                    for j in range(axis, len(ret.input_orders[i])):
                        l.append((len(l), ret.input_orders[i][j][1]))
                else:
                    l.append((len(l), outer_loop))
                    for j in range(len(ret.input_orders[i])):
                        l.append((len(l), ret.input_orders[i][j][1]))

            def action(n, res):
                if isinstance(n, asg.Tensor) and not 'scope' in n.attr:
                    res.extend(n.compute)
                    if True:#helpers.depend_on_item(n, outer_loop.iterate): # TODO: (Yihua) check if n depends on items, if not we don't need to put it in the loop body
                        for nn in n.compute:
                            nn.attr['parent_loop'] = outer_loop
                        outer_loop.body.append(n.compute)
                        n.attr['scope'] = outer_loop.body

            t = helpers.ASGTraversal(action)
            ret_compute = t(ret)

            size = helpers.get_ir_of_size(node.ref_size)
            node.eval = ir.Ndarray(ret.eval.dtype, size)
            node.decl.append(ir.Decl(node.eval))

            res = bind(node.eval, [outer_loop.iterate])
            ret_eval = ret.attr['eval'] if 'eval' in ret.attr else ret.eval
            helpers.replace_all_ref(ret_compute, ret_eval, res)
            helpers.remove_decl(ret, ret_eval)

            # if there is no compute in the func, we simply assign the result to itself, so that later the lhs of the assignment will be changed to the output array
            if len(ret_compute) == 0:
                ret_compute.append(ir.Assignment(res, ret.eval))
                for nn in ret_compute:
                    nn.attr['parent_loop'] = outer_loop
                outer_loop.body.extend(ret_compute)

            node.compute = [outer_loop]

            # ret.eval is removed from the decl
            node.decl = [d for d in node.decl if not helpers.same_object(d.dobject, ret.eval)]

            if cond != None:
                counter = node.operators[-1].eval
                counter.attr['loop'] = outer_loop
                node.compute = [ir.Assignment(counter, 0)] + node.compute
                outer_loop.body.append(ir.Assignment(counter, 1, '+'))
                assert type(ret_compute[-1]) in (ir.Loop, ir.Assignment)
                l = ret_compute[-1]
                while (type(l) == ir.Loop):
                    l = l.body[-1]
                helpers.rebind_iterate(l.lhs, outer_loop.iterate, counter)
                node.attr['eval'] = node.eval

                subscripts = [ir.Slice(ir.Literal(0, counter.dtype), counter, ir.Literal(1, counter.dtype))]
                for i in range(1, len(node.ref_size)):
                    op = asg.Const(slice(0, node.ref_size[i], 1), 'slice')
                    gen_ir(op)
                    subscripts.append(op.eval)
                node.eval = bind(node.eval, subscripts)

                node.attr['is_set'] = True
            # TODO: (Yihua) I don't remember how this is implemented. What is 'is_set' used for? Yihua can you add some comments to explain it.
            elif 'is_set' in node.operators[1].attr:
                size[primary_axis] = node.operators[1].eval.size[primary_axis]
                node.attr['is_set'] = True

            node.output_order = [(0, outer_loop)]
            outer_loop.attr['output_axis'] = 0
            if hasattr(ret, 'output_order'):
                for i in range(len(ret.output_order)):
                    node.output_order.append((i + 1, ret.output_order[i][1]))
                    ret.output_order[i][1].attr['output_axis'] = i + 1


        elif node.op_type == 'reduce':
            gen_ir(node.operators[0])  # input data
            gen_ir(node.operators[3])  # axis
            axis = node.operators[3].eval.val

            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = ir.Ndarray(node.dtype, size)
            else:
                node.eval = ir.Scalar(node.dtype)

            gen_ir(node.operators[2])  # init
            # the decl of node.eval should be added to the init
            node.operators[2].decl.append(ir.Decl(node.eval))


            reduce_loop = ir.Loop(0, node.operators[0].eval.size[axis], 1, [], 'reduction')

            subscripts = []

            data = node.operators[0]
            n = num_unbind(data.eval)

            for i in range(axis):
                op = asg.Const(slice(0, data.ref_size[i], 1), 'slice')
                gen_ir(op)
                subscripts.append(op.eval)

            subscripts.append(reduce_loop.iterate)

            for i in range(axis+1, len(data.ref_size)):
                op = asg.Const(slice(0, data.ref_size[i], 1), 'slice')
                gen_ir(op)
                subscripts.append(op.eval)

            real_subscripts = resolve_view(data, subscripts)

            item1 = node.operators[4]
            item2 = node.operators[5]
            item1.eval = node.eval
            item2.eval = data.eval

            item2.eval = bind(data.eval, real_subscripts)

            if 'dim_map' in data.attr and 'size_map' in data.attr:
                dim_map = [data.attr['dim_map'][ii] for ii in range(len(subscripts)) if
                           type(subscripts[ii]) == ir.Slice]
                size_map = [data.attr['size_map'][ii] for ii in range(len(subscripts)) if
                            type(subscripts[ii]) == ir.Slice]
                item2 = node.operators[5]
                tmp = list(range(len(real_subscripts)))
                j = 0
                for k in range(len(real_subscripts)):
                    if type(real_subscripts[k]) == ir.Slice:
                        tmp[k] = j
                        j += 1
                    else:
                        tmp[k] = None
                item2.attr['dim_map'] = []
                item2.attr['size_map'] = []
                for k in range(len(dim_map)):
                    d = dim_map[k]
                    if d == -1:
                        item2.attr['dim_map'].append(-1)
                        item2.attr['size_map'].append(size_map[k])
                    else:
                        if tmp[d] is not None:
                            item2.attr['dim_map'].append(tmp[d])
                            item2.attr['size_map'].append(size_map[k])


            ret = node.operators[-1]
            gen_ir(ret)

            # TODO: get the input orders


            # 'compute' is the loop body where init should be inserted
            compute = ret.output_order[-1][1].body if len(ret.output_order) > 0 else ret.compute
            if len(compute) == 0:
                compute.append(ir.Assignment(node.eval, ret.eval))
            # put the statements in 'compute' into the reduce_loop
            reduce_loop.body = compute[:]
            compute.clear()

            # merge init into compute
            init = node.operators[2].output_order[-1][1].body if len(node.operators[2].output_order) > 0 else \
            node.operators[2].compute
            for i in range(len(node.operators[2].output_order)):
                helpers.rebind_iterate(init, node.operators[2].output_order[i][1].iterate, ret.output_order[i][1].iterate)
                node.output_order.append((i, ret.output_order[i][1]))
                ret.output_order[i][1].attr['output_axis'] = i
            compute.extend(init)
            node.operators[2].compute.clear()
            compute.append(reduce_loop)

            # replace ret.eval with node.eval and remove decl of ret.eval
            helpers.replace_all_ref(ret.compute, ret.eval, node.eval)
            ret.decl = [d for d in ret.decl if not helpers.same_object(d.dobject, ret.eval)]



        elif node.op_type == 'aggr':
            gen_ir(node.operators[0])  # input tensor
            gen_ir(node.operators[3])  # indices
            gen_ir(node.operators[4])  # axis
            axis = node.operators[4].eval.val
            size = helpers.get_ir_of_size(node._size())
            node.eval = ir.Ndarray(node.dtype, size)
            gen_ir(node.operators[2])  # init
            node.operators[2].decl.append(ir.Decl(node.eval))

            # compute
            outer_loop = ir.Loop(0, node.operators[0].eval.size[axis], 1, [], 'reduction')

            item1 = node.operators[6]
            item2 = node.operators[7]
            item1.eval = ir.Indexing(node.eval, ir.Indexing(node.operators[3].eval, outer_loop.iterate))
            item2.eval = node.operators[0].eval
            for i in range(axis):
                item2.eval = ir.Indexing(item2.eval, ir.Literal(-1, 'int'))
            item2.eval = ir.Indexing(item2.eval, outer_loop.iterate)
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            gen_ir(ret)

            def action(node, res):
                if isinstance(node, asg.Tensor):
                    res.extend(node.compute)
                    node.compute.clear()

            ret_compute = helpers.ASGTraversal(action)(ret)
            
            for nn in ret_compute:
                nn.attr['parent_loop'] = outer_loop
            outer_loop.body.extend(ret_compute)
            node.compute.append(outer_loop)

            replace_output(node.compute, ret.eval, item1.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

            node.output_order = [(0, outer_loop)]
            for i in range(len(ret.output_order)):
                node.output_order.append((i + 1, ret.output_order[i][1]))
                ret.output_order[i][1].attr['output_axis'] = i + 1

        elif node.op_type == 'inline':
            src = node.operators[0]
            num_output = node.operators[1].val
            outputs_keyvalue = []
            inputs_keyvalue = []
            for i in range(2, len(node.operators), 2):
                gen_ir(node.operators[i+1])
                if i<=num_output*2:
                    gen_ir(node.operators[i+1])
                    outputs_keyvalue.append((node.operators[i], node.operators[i+1].eval))
                else:
                    gen_ir(node.operators[i+1])
                    inputs_keyvalue.append((node.operators[i], node.operators[i+1].eval))
            node.eval = node.operators[3].eval
            node.compute = [ir.Code(src, dict(outputs_keyvalue), dict(inputs_keyvalue))]

        elif node.op_type == 'size':
            gen_ir(node.operators[0])
            gen_ir(node.operators[1])

            axis = node.operators[1].eval.val
            node.eval = ir.Scalar(node.operators[0]._size()[0].dtype)
            node.decl = [ir.Decl(node.eval)]
            node.compute = [ir.Assignment(node.eval, node.operators[0].eval.size[axis])]

        elif node.op_type == 'mklist':
            for opr in node.operators:
                gen_ir(opr)


        # TODO: (Lihan) what does this do? what is the storage attribute?
        # storage attr stores all the other representations of current node.eval, it is used in parallelize.py
        if node.eval != None:
            node.eval.attr['storage'] = []






    return node
