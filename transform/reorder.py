import codegen.cpu
from ir import *
from asg import *
from helpers import rebind_iterate


def output_reorder(node, dim_order, tile_size):
    assert isinstance(node, TensorOp)
    assert node.op_type in arith_op or node.op_type in math_op or node.op_type in ('einsum', 'setval')
    assert len(node.compute) == 1
    assert len(dim_order) == len(tile_size) and len(dim_order) > 0
    loop = node.compute[0]

    for i in range(len(node._size())):
        # add the nontiled dimensions
        if not i in dim_order:
            dim_order.append(i)
            tile_size.append(0)

    assert sorted(dim_order) == list(range(len(dim_order)))

    loop_nest = []
    l = loop
    for i in range(len(dim_order)):
        loop_nest.append(l)
        l = loop.body[0]

    tile_loops = []
    reorder_nest = []
    new_indices = []
    output_order = []
    for i in range(len(dim_order)):
        l = loop_nest[dim_order[i]]
        if tile_size[i] > 0:
            tl = Loop(l.start, l.end, tile_size[i], [])
            if tile_size[i] == 1:
                new_indices.append(tl.iterate)
        else:
            tl = None
        tile_loops.append(tl)
        reorder_nest.append(l)
        if tl != None:
            output_order.append((dim_order[i], tl))

    for i in range(len(reorder_nest)):
        ts = tile_size[i]
        if ts > 1:
            tl = tile_loops[i]
            new_l = Loop(tl.iterate, Expr(Expr(tl.iterate, ts, '+'), l.end,'smaller'), l.step, [])
        elif ts == 1:
            new_l = None
        else:
            l = reorder_nest[i]
            new_l = Loop(l.start, l.end, l.step, [])
        if new_l != None:
            tile_loops.append(new_l)
            new_indices.append(new_l.iterate)
            output_order.append((dim_order[i], new_l))

    tile_loops = [l for l in tile_loops if l != None]

    for i in range(len(tile_loops)-1, 0, -1):
        tile_loops[i-1].body.append(tile_loops[i])

    body = loop_nest[-1].body

    for i in range(len(dim_order)):
        old_itr = reorder_nest[i].iterate
        new_itr = new_indices[i]
        rebind_iterate(body, old_itr, new_itr)

    tile_loops[-1].body.extend(body)
    node.compute = [tile_loops[0]]
    node.output_order = output_order




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
    TS2 = 10

    output_reorder(ir1, [0, 1], [TS1, TS2])
    print(res1.output_order)
    code = codegen.cpu.print_cpp(ir1)
    print(code)

    # res2 = res1 + C
    # ir2 = res2._gen_ir()
    # print(res2.input_orders)
    # print(res2.output_order)



