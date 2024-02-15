import copy
import ir
import asg
import helpers


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
            assert type(index.idx) in (ir.Scalar, ir.Literal)
            if type(index.idx) == ir.Scalar or (type(index.idx) == ir.Literal and index.idx.val != -1):
                continue
            idx = subscripts[j]
            if type(idx) in (ir.Scalar, ir.Literal, ir.Indexing):
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
        if type(idx) in (ir.Scalar, ir.Literal, ir.Indexing):
            new_index = ir.Indexing(new_index, idx)
        elif type(idx) in (ir.Ndarray, ir.Slice):
            new_index = ir.Indexing(new_index, ir.Indexing(idx, ir.Literal(-1, 'int')))
        else:
            raise TypeError('incorrect idx type!')
        # new_index.attr.update(attrs[j])
        j += 1
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
            else:  # otherwise, it is a scalar
                size = []
                node.eval = ir.Scalar(node.dtype)
            node.decl = [ir.Decl(node.eval)]

            left_levels = len(node.operators[0]._size())
            right_levels = len(node.operators[1]._size())
            max_levels = max(left_levels, right_levels)
            assert max_levels == len(size)

            lhs = node.operators[0].eval
            rhs = node.operators[1].eval
            res = node.eval
            compute = node.compute

            for level in range(max_levels):

                # handle out of bound slicing
                left_slice = get_slice(lhs)
                right_slice = get_slice(rhs)
                left_attr = {}
                if left_slice != None and type(left_slice.start) == ir.Literal:
                    if left_slice.start.val < 0:
                        left_ofs = -left_slice.start.val
                        left_attr['slice_ofs'] = left_ofs
                    else:
                        left_ofs = 0
                else:
                    left_ofs = 0
                right_attr = {}
                if right_slice != None and type(right_slice.start) == ir.Literal:
                    if right_slice.start.val < 0:
                        right_ofs = -right_slice.start.val
                        right_attr['slice_ofs'] = right_ofs
                    else:
                        right_ofs = 0
                else:
                    right_ofs = 0

                pre_loop = ir.Loop(0, size[level], 1, [])
                loop_ofs = max(left_ofs, right_ofs)
                if loop_ofs > 0:
                    pre_loop.attr['loop_ofs'] = loop_ofs

                if level < left_levels:
                    lhs = bind(lhs, [pre_loop.iterate], [left_attr])
                    node.input_orders[0].append((level, pre_loop))
                if level < right_levels:
                    rhs = bind(rhs, [pre_loop.iterate], [right_attr])
                    node.input_orders[1].append((level, pre_loop))
                res = bind(res, [pre_loop.iterate])
                node.output_order.append((level, pre_loop))
                pre_loop.attr['output_axis'] = level
                compute.append(pre_loop)
                compute = pre_loop.body
            assign = ir.Assignment(res, ir.Expr(lhs, rhs, op))
            assign.attr['parent_loop'] = pre_loop
            compute.append(assign)

        elif node.op_type in asg.math_op:
            gen_ir(node.operators[0])

            if len(node._size()) > 0:
                size = helpers.get_ir_of_size(node._size())
                node.eval = ir.Ndarray(node.dtype, size)
            else:
                size = []
                node.eval = ir.Scalar(node.dtype)

            node.decl = [ir.Decl(node.eval)]

            res = node.eval
            val = node.operators[0].eval
            levels = len(size)
            compute = node.compute

            for level in range(levels):
                slice = get_slice(val)
                attr = {}
                if slice != None and type(slice.start) == ir.Literal:
                    if slice.start.val < 0:
                        ofs = -slice.start.val
                        attr['slice_ofs'] = ofs
                    else:
                        ofs = 0
                else:
                    ofs = 0

                pre_loop = ir.Loop(0, size[level], 1, [])
                if ofs > 0:
                    pre_loop.attr['loop_ofs'] = ofs

                val = bind(val, [pre_loop.iterate], [attr])
                node.input_orders[0].append((level, pre_loop))
                res = bind(res, [pre_loop.iterate])
                node.output_order.append((level, pre_loop))
                pre_loop.attr['output_axis'] = level
                compute.append(pre_loop)
                compute = pre_loop.body

            compute.append(ir.Assignment(res, ir.Math(val, node.op_type)))

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
                node.operators[1].eval = node.eval
                replace_output(node.operators[1].compute, node.operators[1].eval, node.eval)


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
            for i in input1:
                op1 = bind(op1, [all_loops[mapping[i]].iterate])

            if node.operators[1] != None:
                op2 = node.operators[1].eval
                for i in input2:
                    op2 = bind(op2, [all_loops[mapping[i]].iterate])
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


        elif node.op_type == 'index':
            gen_ir(node.operators[0])
            subscripts = []
            for op in node.operators[1:]:
                gen_ir(op)
                subscripts.append(op.eval)

            node.eval = bind(node.operators[0].eval, subscripts)
            for key in node.operators[0].eval.attr:
                node.eval.attr[key] = node.operators[0].eval.attr[key]

        elif node.op_type == 'apply':

            # operators: func, data (node.nparams), axis (node.nparams), out_ofs, cond, items (node.nparams), ret
            for i in range(1, 3 + 2 * node.nparams):
                if node.operators[i] != None:
                    gen_ir(node.operators[i])

            primary_axis = node.operators[1 + node.nparams].eval.val

            # this is the loop that iterates over the axis of the primary (first) tensor input
            cond = node.operators[2 + 2 * node.nparams]
            if cond == None:
                outer_loop = ir.Loop(0, node.operators[1].eval.size[primary_axis], 1, [])
            else:
                outer_loop = ir.FilterLoop(0, node.operators[1].eval.size[primary_axis], 1,
                                        cond.eval, [], [])
                # gen ir for the counter
                gen_ir(node.operators[-1])


            nn = []
            for i in range(node.nparams):
                item = node.operators[3 + 2 * node.nparams + i]
                item.eval = node.operators[1 + i].eval
                axis = node.operators[1 + node.nparams + i].eval.val
                n = num_unbind(item.eval)
                nn.append(n)
                for i in range(n, axis):
                    item.eval = ir.Indexing(item.eval, ir.Literal(-1, 'int'))
                if axis >= n:
                    item.eval = ir.Indexing(item.eval, outer_loop.iterate)
                else:
                    item.eval = bind(item.eval, [outer_loop.iterate])

            # since input items of func has been generated and indexed, we can generate the IR of the func
            ret = node.operators[-2]
            gen_ir(ret)

            # get the input orders
            for i in range(min(len(ret.input_orders), node.nparams)):
                n = nn[i]
                l = node.input_orders[1 + i]
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
                    if helpers.depend_on_item(n, outer_loop.iterate): # TODO: check if n depends on items, if not we don't need to put it in the loop body
                        for nn in n.compute:
                            nn.attr['parent_loop'] = outer_loop
                        outer_loop.body.append(n.compute)
                        n.attr['scope'] = outer_loop.body

            t = helpers.ASGTraversal(action)
            ret_compute = t(ret)

            size = helpers.get_ir_of_size(node._size())
            node.eval = ir.Ndarray(ret.eval.dtype, size)
            node.decl.append(ir.Decl(node.eval))

            out_ofs = node.operators[1 + 2 * node.nparams]
            res = bind(node.eval, [outer_loop.iterate]) if out_ofs == None else node.eval
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

            # if there is an offset for output storage
            if out_ofs != None:
                assert type(ret_compute[-1]) in (ir.Loop, ir.Assignment)
                l = ret_compute[-1]
                while (type(l) == ir.Loop):
                    l = l.body[-1]
                # But the index to the node.eval in res is incorrect, we need to change it according to the offset
                helpers.rebind_iterate(l.lhs, ret_compute[-1].iterate,
                               ir.Expr(ir.Indexing(out_ofs.eval, outer_loop.iterate), ret_compute[-1].iterate, '+'))
            # ret.eval is removed from the decl
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

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
                node.eval = bind(node.eval, [ir.Slice(ir.Literal(0, 'int'), counter, ir.Literal(1, 'int'))])
                node.attr['is_set'] = True
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
            # TODO: add input_orders for reduce, and aggr
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

            outer_loop = ir.Loop(0, node.operators[0].eval.size[axis], 1, [], 'reduction')

            item1 = node.operators[4]
            item2 = node.operators[5]
            item1.eval = node.eval
            item2.eval = node.operators[0].eval
            n = num_unbind(item2.eval)
            for i in range(n, axis):
                item2.eval = ir.Indexing(item2.eval, ir.Literal(-1, 'int'))
            if axis > n:
                item2.eval = ir.Indexing(item2.eval, outer_loop.iterate)
            else:
                item2.eval = bind(item2.eval, [outer_loop.iterate])
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            gen_ir(ret)

            compute = ret.output_order[-1][1].body if len(ret.output_order) > 0 else ret.compute
            outer_loop.body = compute[:]
            compute.clear()

            # merge init into node.compute
            init = node.operators[2].output_order[-1][1].body if len(node.operators[2].output_order) > 0 else \
            node.operators[2].compute
            # assert len(node.operators[2].output_order) == len(ret.output_order)
            for i in range(len(node.operators[2].output_order)):
                # assert has_same_iteration_space(node.operators[2].output_order[i][1], ret.output_order[i][1])
                helpers.rebind_iterate(init, node.operators[2].output_order[i][1].iterate, ret.output_order[i][1].iterate)
                node.output_order.append((i, ret.output_order[i][1]))
                ret.output_order[i][1].attr['output_axis'] = i
            compute.extend(init)
            node.operators[2].compute.clear()
            compute.append(outer_loop)

            def action(node, res):
                if isinstance(node, asg.Tensor):
                    res.extend(node.compute)
                    node.compute.clear()

            t = asg.ASGTraversal(action)
            ret_compute = t(ret)

            node.compute.extend(ret_compute)

            replace_output(node.compute, ret.eval, node.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]


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

            t = asg.ASGTraversal(action)
            ret_compute = t(ret)
            
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
            keyvalue = []
            for i in range(2, len(node.operators), 2):
                gen_ir(node.operators[i])
                keyvalue.append((node.operators[i-1], node.operators[i].eval))

            node.eval = node.operators[2].eval
            node.compute = [ir.Code(src, keyvalue[0], dict(keyvalue[1:]))]

        elif node.op_type == 'size':
            gen_ir(node.operators[0])
            gen_ir(node.operators[1])

            axis = node.operators[1].eval.val
            node.eval = ir.Scalar('int')
            node.decl = [ir.Decl(node.eval)]
            node.compute = [ir.Assignment(node.eval, node.operators[0].eval.size[axis])]







    return node
