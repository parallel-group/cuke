import codegen.cpu
from ir import *
from asg import *
from helpers import rebind_iterate, IRTraversal


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

def split(node, tile_size, axis):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval']
    assert type(tile_size) == int and tile_size > 0
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
        tl = Loop(loop.start, loop.end, tile_size, [])
        tl.attr['output_axis'] = loop.attr['output_axis']
        new_loops.append((axis, tl))
        if tile_size > 1:
            new_l = Loop(tl.iterate, Expr(Expr(tl.iterate, tile_size, '+'), loop.end,'smaller'), loop.step, [])
            new_l.attr['output_axis'] = tl.attr['output_axis']
            tl.body.append(new_l)
            new_loops.append((axis, new_l))

        body = loop.body[:]
        rebind_iterate(body, loop.iterate, new_loops[-1][1].iterate)
        new_loops[-1][1].body.extend(body)
        _replace_loop(node.compute, loop, new_loops[0][1])
        node.output_order[order_idx] = (axis, new_loops[-1][1])
        if len(new_loops) > 1:
            for i in range(len(new_loops)-2, -1, -1):
                node.output_order.insert(order_idx, (axis, new_loops[i][1]))


if __name__ == "__main__":
    A = Tensor('a', (10, 20))
    B = Tensor('b', (20, 30))
    C = Tensor('c', (10, 30))
    res1 = A @ B
    ir1 = res1._gen_ir()
    print(res1.output_order)
    code = codegen.cpu.print_cpp(ir1)
    print(code)

    TS1 = 5
    split(ir1, TS1, 0)
    print(res1.output_order)
    code = codegen.cpu.print_cpp(ir1)
    print(code)

    split(ir1, 2, 1)
    print(res1.output_order)
    code = codegen.cpu.print_cpp(ir1)
    print(code)
