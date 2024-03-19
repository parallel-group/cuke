import codegen.cpu
from ir import *
from asg import *
from helpers import rebind_iterate, IRTraversal, flatten, get_loops_at_level
from asg2ir import gen_ir

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

# TODO: add support for filterloop
def split_loop(node, bsize, idx: list|tuple):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval', 'norm', 'aggr']
    assert type(bsize) == int and bsize > 0

    scope = flatten(node.compute)
    for i in idx:
        loop = None
        j = -1
        for s in scope:
            if isinstance(s, Loop):
                j += 1
                if j == i:
                    loop = s
                    break
        if loop != None:
            scope = flatten(loop.body)

    # if loop != None and 'reduction' not in loop.attr:
    if loop != None:
        new_loops = []
        tl = Loop(loop.start, loop.end, bsize, [])
        tl.attr['ptype'] = loop.attr['ptype']
        if 'parent_loop' in loop.attr:
            tl.attr['parent_loop'] = loop.attr['parent_loop']
        axis = None
        order_idx = None
        if 'output_axis' in loop.attr:
            axis = tl.attr['output_axis'] = loop.attr['output_axis']
            for i in range(len(node.output_order)):
                if node.output_order[i][0] == axis and node.output_order[i][1] == loop:
                    order_idx = i
                    break

        new_loops.append((axis, tl))
        if bsize > 1:
            stop = Expr(Expr(tl.iterate, bsize, '+'), loop.end, 'smaller')
            stop.attr['loop'] = tl
            new_l = Loop(tl.iterate, stop, loop.step, [])
            new_l.attr['ptype'] = loop.attr['ptype']
            if 'output_axis' in loop.attr:
                new_l.attr['output_axis'] = loop.attr['output_axis']
            new_l.attr['parent_loop'] = tl
            tl.body.append(new_l)
            new_loops.append((axis, new_l))

        body = loop.body[:]
        rebind_iterate(body, loop.iterate, new_loops[-1][1].iterate)
        for i in body:
            if isinstance(i, Loop):
                i.attr['parent_loop'] = new_loops[-1][1]
            elif isinstance(i, list|tuple):
                for j in i:
                    j.attr['parent_loop'] = new_loops[-1][1]
        new_loops[-1][1].body.extend(body)
        _replace_loop(node.compute, loop, new_loops[0][1])
        if order_idx != None:
            assert axis != None
            node.output_order[order_idx] = (axis, new_loops[-1][1])
            if len(new_loops) > 1:
                for i in range(len(new_loops) - 2, -1, -1):
                    node.output_order.insert(order_idx, (axis, new_loops[i][1]))
        # print(codegen.gpu.to_string(node.compute), node.output_order)
        else:
            node.output_order.extend(new_loops)
        new_loops = [(axis, new_loops[i][1]) for i in range(len(new_loops))]
        for iorder in node.input_orders:
            iidx = None
            for i in range(len(iorder)):
                if iorder[i][1] == loop:
                    iidx = i
                    break
            if iidx != None:
                iorder[iidx:iidx+1] = new_loops
        
        for i in range(len(new_loops)):
            if i>0:
                new_loops[i][1].attr['parent_loop'] = new_loops[i-1][1]




def split_level(node, bsize, level):
    loops = []
    get_loops_at_level(node.compute, level, [], loops)
    for l in loops:
        split_loop(node, bsize, l)



def split_axis(node, bsize, axis):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval']
    assert type(bsize) == int and bsize > 0
    assert type(axis) == int and axis >= 0

    loop = None
    order_idx = None
    for i in range(len(node.output_order)):
        if node.output_order[i][0] == axis:
            loop = node.output_order[i][1]
            order_idx = i
            break

    if loop != None:
        new_loops = []
        tl = Loop(loop.start, loop.end, bsize, [])
        tl.attr['output_axis'] = loop.attr['output_axis']
        new_loops.append((axis, tl))
        if bsize > 1:
            stop = Expr(Expr(tl.iterate, bsize, '+'), loop.end, 'smaller')
            stop.attr['loop'] = tl
            new_l = Loop(tl.iterate, stop, loop.step, [])
            new_l.attr['output_axis'] = tl.attr['output_axis']
            tl.body.append(new_l)
            new_loops.append((axis, new_l))

        body = loop.body[:]
        rebind_iterate(body, loop.iterate, new_loops[-1][1].iterate)
        new_loops[-1][1].body.extend(body)
        _replace_loop(node.compute, loop, new_loops[0][1])
        node.output_order[order_idx] = (axis, new_loops[-1][1])
        num_new_loops = 1
        if len(new_loops) > 1:
            for i in range(len(new_loops)-2, -1, -1):
                node.output_order.insert(order_idx, (axis, new_loops[i][1]))
                num_new_loops += 1

        for iorder in node.input_orders:
            iidx = None
            for i in range(len(iorder)):
                if iorder[i][1] == loop:
                    iidx = i
                    break
            if iidx != None:
                iorder[iidx:iidx+1] = node.output_order[order_idx:order_idx+num_new_loops]






if __name__ == "__main__":
    A = Tensor((10, 20), name='A')
    B = Tensor((20, 30), name='B')
    C = Tensor((10, 30), name='C')
    res = A @ B
    ir = gen_ir(res)

    print(get_loop_nests(ir.compute))
    code = codegen.cpu.print_cpp(ir)
    print(code)

    TS1 = 5
    split_level(ir, TS1, 0)
    # split_loop(ir, TS1, [0])
    code = codegen.cpu.print_cpp(ir)
    print(code)
    print(ir.output_order)

    # TS2 = 10
    # split_loop(ir, TS2, [0,0,0])
    # print(get_loop_nests(ir.compute))
    # code = codegen.cpu.print_cpp(ir)
    # print(code)
    #
    # print(ir.output_order)
