import torch

import pycuke.helpers as helpers
import pycuke.transform as transform
from pycuke.asg import Tensor, Var, apply, bigger, setval, einsum
from pycuke.asg2ir import gen_ir
import pycuke.codegen as codegen
import pycuke.run as run


def test1():
    A = Tensor((10, 20))
    B = Tensor((10, 20))
    res = A + B - 1
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20)
    B = torch.rand(10, 20)

    d = run.cpu.compile_and_run(code, A, B)
    print(torch.equal(A + B - 1, d))


def test2():

    n = Var('int')
    m = Tensor((n, 2))
    i = Var('int')
    t = Tensor(m[i]._size())
    res = m[i] + t
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    n = 10
    m = torch.rand(n, 2)
    i = 5
    t = torch.rand(m[i].shape)

    d = run.cpu.compile_and_run(code, n, m, i, t)
    print(torch.equal(m[i] + t, d))



def test3():
    A = Tensor((10, 2, 2))
    i = Var('int')
    j = Var('int')
    t = Tensor(A[i][j]._size())

    ast = A[i][j] + t

    code = codegen.cpu.print_cpp(gen_ir(ast))
    print(code)

    A = torch.rand(10, 2, 2)
    i = 1
    j = 1
    t = torch.rand(A[i][j].shape)

    d = run.cpu.compile_and_run(code, A, i, j, t)
    print(torch.equal(A[i][j] + t, d))



def test4():
    A = Tensor((10, ))
    i = Var('int')
    t = Var(A.dtype)

    ast = A[i] + t
    code = codegen.cpu.print_cpp(gen_ir(ast))

    A = torch.rand(10)
    i = 1
    t = 2
    d = run.cpu.compile_and_run(code, A, i, t)
    print(d, A[i] + t)

def test5():
    A = Tensor((10, 20))
    B = Tensor((10, 20))
    res = A[1:3][1] + B[2:4][0] - 1
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20)
    B = torch.rand(10, 20)

    d = run.cpu.compile_and_run(code, A, B)
    print(torch.equal(A[1:3][1] + B[2:4][0] - 1, d))

def test6():
    A = Tensor((10, ))
    idx = Tensor((5, ), dtype='int')
    t = Tensor(A[idx]._size())

    res = A[idx] + t

    code = codegen.cpu.print_cpp(gen_ir(res))
    A = torch.rand(10)
    idx = torch.IntTensor(5)
    t = torch.rand(A[idx].shape)
    d = run.cpu.compile_and_run(code, A, idx, t)

    print(d)

    print(torch.equal(d, A[idx] + t))


def test7():
    A = Tensor((10, 10))
    idx = Tensor((5, ), dtype='int')
    t = Tensor(A[idx]._size())

    res = A[idx] + t

    code = codegen.cpu.print_cpp(gen_ir(res))
    A = torch.rand(10, 10)
    idx = torch.IntTensor(5)
    t = torch.rand(A[idx].shape)
    d = run.cpu.compile_and_run(code, A, idx, t)

    print(d)

    print(torch.equal(d, A[idx] + t))

def test8():
    A = Tensor((10, 10))
    idx = Tensor((5, ), dtype='int')

    res = A[0][idx] + 1

    code = codegen.cpu.print_cpp(gen_ir(res))
    A = torch.rand(10, 10)
    idx = torch.IntTensor(5)
    d = run.cpu.compile_and_run(code, A, idx)

    print(d)

    print(torch.equal(d, A[0][idx] + 1))

def test9():
    A = Tensor((10, 10))
    i = Tensor((5, ), dtype='int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[i][j]._size())

    res = A[i][j] + t

    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)
    A = torch.rand(10, 10)
    i = torch.IntTensor(5)
    j = torch.IntTensor(4)
    t = torch.rand(A[i][j].shape)
    d = run.cpu.compile_and_run(code, A, i, j, t)

    print(A[i][j] + t)

    print(torch.equal(d, A[i][j] + t))

def test9_1():
    A = Tensor((10, 10))
    i = Tensor((5, ), dtype='int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[i][:,j]._size())

    res = A[i][:,j] + t

    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)
    A = torch.rand(10, 10)
    i = torch.IntTensor(5)
    j = torch.IntTensor(4)
    t = torch.rand(A[i][:,j].shape)
    d = run.cpu.compile_and_run(code, A, i, j, t)

    print(A[i][:,j] + t)

    print(torch.equal(d, A[i][:,j] + t))

def test10():
    A = Tensor((10, 11, 12))
    i = Tensor((5, ), dtype='int')
    x = Var( 'int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[i][j][x]._size())

    ast = A[i][j][x] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = 2
    j = torch.IntTensor(4)
    t = torch.rand(A[i][j][x].shape)
    d = run.cpu.compile_and_run(code, A, i, j, x, t)

    print(A[i][j][x] + t)

    print(torch.equal(d, A[i][j][x] + t))


def test10_1():
    A = Tensor((10, 11, 12))
    i = Tensor((5, ), dtype='int')
    x = Var('int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[i][:,j][x]._size())

    ast = A[i][:,j][x] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = 2
    j = torch.IntTensor(4)
    t = torch.rand(A[i][:,j][x].shape)
    d = run.cpu.compile_and_run(code, A, i, j, x, t)

    print(A[i][:,j][x] + t)

    print(torch.equal(d, A[i][:,j][x] + t))


def test10_2():
    A = Tensor((10, 11, 12))
    i = Tensor((5, ), dtype='int')
    x = Var('int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[i][:,j][:,x]._size())

    ast = A[i][:,j][:,x] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = 2
    j = torch.IntTensor(4)
    t = torch.rand(A[i][:,j][:,x].shape)
    d = run.cpu.compile_and_run(code, A, i, j, x, t)

    print(A[i][:,j][:,x] + t)

    print(torch.equal(d, A[i][:,j][:,x] + t))


def test10_3():
    A = Tensor((10, 11, 12))
    i = Tensor((5, ), dtype='int')
    x = Var('int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[i][:,j][:,:,x]._size())

    ast = A[i][:,j][:,:,x] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = 2
    j = torch.IntTensor(4)
    t = torch.rand(A[i][:,j][:,:,x].shape)
    d = run.cpu.compile_and_run(code, A, i, j, x, t)

    print(A[i][:,j][:,:,x] + t)
    # print(d)

    print(torch.equal(d, A[i][:,j][:,:,x] + t))

def test10_4():
    A = Tensor((10, 11, 12))
    i = Tensor((5, ), dtype='int')
    x = Tensor((3, ), 'int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[i][:,j][:,:,x]._size())

    ast = A[i][:,j][:,:,x] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = torch.IntTensor(3)
    j = torch.IntTensor(4)
    t = torch.rand(A[i][:,j][:,:,x].shape)
    d = run.cpu.compile_and_run(code, A, i, j, x, t)

    print(A[i][:,j][:,:,x] + t)
    # print(d)

    print(torch.equal(d, A[i][:,j][:,:,x] + t))



def test10_5():
    A = Tensor((10, 11, 12))
    i = Tensor((5, ), dtype='int')
    x = Tensor((3, ), 'int')
    j = Tensor((4, ), dtype='int')
    t = Tensor(A[:,i][:,j][:,:,x]._size())

    ast = A[:, i][:,j][:,:,x] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = torch.IntTensor(3)
    j = torch.IntTensor(4)
    t = torch.rand(A[:,i][:,j][:,:,x].shape)
    d = run.cpu.compile_and_run(code, A, i, j, x, t)

    print(A[:,i][:,j][:,:,x] + t)
    # print(d)

    print(torch.equal(d, A[:,i][:,j][:,:,x] + t))

def test11():
    A = Tensor('a', (10, 11, 12))
    i = Tensor('i', (5, ), dtype='int')
    x = Var('x', 'int')
    y = Var('y', 'int')
    j = Tensor('j', (y, ), dtype='int')
    t = Tensor('t', A[:,i][x][:,j]._size())

    ast = A[:,i][x][:,j] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = 2
    y = 3
    j = torch.IntTensor(y)
    t = torch.rand(A[:,i][x][:,j].shape)
    d = run.cpu.compile_and_run(code, y, A, i, x, j, t)

    print(A[:,i][x][:,j] + t)

    print(torch.equal(d, A[:,i][x][:,j] + t))


def test12():
    s1 = Var('s1', 'int')
    s2 = Var('s2', 'int')
    s3 = Var('s3', 'int')
    A = Tensor('a', (s1, s2, s3))
    b1 = Var('b1', 'int')
    v1 = Tensor('v1', (b1, ), dtype='int')
    b2 = Var('b2', 'int')
    v2 = Tensor('v2', (b2, ), dtype='int')
    x = Var('x', 'int')
    t = Tensor('t', A[v1][:,v2][:,:,x]._size())

    ast = A[v1][:,v2][:,:,x] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    s1 = 10
    s2 = 20
    s3 = 30
    A = torch.rand(s1, s2, s3)
    b1 = 3
    v1 = torch.IntTensor(b1)
    b2 = 4
    v2 = torch.IntTensor(b2)
    x = 2
    t = torch.rand(A[v1][:,v2][:,:,x].shape)
    d = run.cpu.compile_and_run(code, b1, b2, s3, s2, s1, A, v1, v2, x, t)

    print(A[v1][:,v2][:,:,x] + t)

    print(torch.equal(d, A[v1][:,v2][:,:,x] + t))

def test13():
    s1 = Var('s1', 'int')
    s2 = Var('s2', 'int')
    A = Tensor('a', (s1, s2))
    b1 = Var('b1', 'int')
    v1 = Tensor('v1', (b1, ), dtype='int')
    v2 = Tensor('v2', (b1, ), dtype='int')
    x = Var('x', 'int')
    t = Tensor('t', A[v1+v2][:,x]._size())

    ast = A[v1+v2][:,x] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    s1 = 10
    s2 = 20
    A = torch.rand(s1, s2)
    b1 = 3
    v1 = torch.IntTensor(b1)
    v2 = torch.IntTensor(b1)
    x = 2
    t = torch.rand(A[v1+v2][:,x].shape)
    d = run.cpu.compile_and_run(code, b1, s2, s1, A, v1, v2, x, t)

    print(A[v1+v2][:,x] + t)

    print(torch.equal(d, A[v1+v2][:,x] + t))


def test13_1():
    s1 = Var('s1', 'int')
    s2 = Var('s2', 'int')
    A = Tensor('a', (s1, s2))
    b1 = Var('b1', 'int')
    v1 = Tensor('v1', (b1, ), dtype='int')
    v2 = Tensor('v2', (b1, ), dtype='int')
    x = Var('x', 'int')
    t = Tensor('t', A[v1+v2][x]._size())

    ast = A[v1+v2][x] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    s1 = 10
    s2 = 20
    A = torch.rand(s1, s2)
    b1 = 3
    v1 = torch.IntTensor(b1)
    v2 = torch.IntTensor(b1)
    x = 2
    t = torch.rand(A[v1+v2][x].shape)
    d = run.cpu.compile_and_run(code, s2, b1, s1, A, v1, v2, x, t)

    print(A[v1+v2][x] + t)

    print(torch.equal(d, A[v1+v2][x] + t))


def test14():
    s1 = Var('s1', 'int')
    A = Tensor('A', (s1, ), dtype='int')
    B = Tensor('B', (A[0], A[1]))
    C = Tensor('C', B._size())

    ast = B + C
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    s1 = 10
    A = torch.randint(5, 10, (s1, ), dtype=torch.int)
    B = torch.rand(A[0], A[1])
    C = torch.rand(B.shape)

    print(B+C)

    d = run.cpu.compile_and_run(code, s1, A, B, C)


    print(torch.equal(d, B+C))


def test15():
    A = Tensor((100, ))
    B = Tensor((100, ))

    ast = A[1:10] + A[0:9] + B[-1:8]
    # print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)





def test16():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, 20))

    ast = A[1:10] + B[1:10]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100, 20)
    B = torch.rand(100, 20)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:10] + B[1:10])
    print(torch.equal(A[1:10] + B[1:10], d))


def test17():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, 20))

    ast = A[1:10][2:4] + B[1:10][2:4]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    A = torch.rand(100, 20)
    B = torch.rand(100, 20)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:10][2:4] + B[1:10][2:4])
    print(d)
    print(torch.equal(A[1:10][2:4] + B[1:10][2:4], d))


def test18():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, ), dtype='int')

    ast = A[1:30][B[2:4]] + A[1:30][B[1:3]]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100, 20)
    B = torch.randint(0, 20, (30, )).to(torch.int32)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:30][B[2:4]] + A[1:30][B[1:3]])
    print(d)
    print(torch.equal(A[1:30][B[2:4]] + A[1:30][B[1:3]], d))



def test18_1():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, ), dtype='int')

    ast = A[1:30][:,B[2:4]] + A[1:30][:,B[1:3]]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100, 20)
    B = torch.randint(0, 20, (30, )).to(torch.int32)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:30][:,B[2:4]] + A[1:30][:,B[1:3]])
    print(d)
    print(torch.equal(A[1:30][:,B[2:4]] + A[1:30][:,B[1:3]], d))

def test19():
    nnodes = 100
    nedges = 300
    rowptr = Tensor('rowptr', (nnodes + 1, ), dtype='int')
    colidx = Tensor('colidx', (nedges, ), dtype='int')
    edge_list = Tensor('edge_list', (10, 2), dtype='int')
    ast = colidx[rowptr[edge_list[0][0]]:rowptr[edge_list[0][1]]] + 1

    print(helpers.get_input_nodes(ast))
    code = codegen.cpu.print_cpp(gen_ir(ast))
    print(code)


def test20():
    A = Tensor('A', (100, ), dtype='float')
    x = Var('x', dtype='float')
    res = A + 10
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


# TODO
def test21():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, 20))

    ast = A[-1:8][:,10:20] + B[1:10][:,-2:8]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)





def test_math1():
    input = Tensor('input', (50, 32), dtype='float')
    res = input[0].abs()
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def test_math2():
    input = Tensor('input', (50, 32), dtype='float')
    input = setval(input, 0)
    res = input[0].abs()
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)




def apply_test1():
    num_edges = 20
    length = 50
    rowidx = Tensor((num_edges,), dtype='int', name='rowidx')
    colidx = Tensor((num_edges,), dtype='int', name='colidx')
    edge_idx = Tensor((length,), dtype='int', name='edge_idx')


    def apply_func(edge_id):
        v0 = rowidx[edge_id]
        v1 = colidx[edge_id] + 1
        return v0 + v1

    res = edge_idx.apply(apply_func)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    edge_idx = torch.randint(0, num_edges, (length,)).to(torch.int32)
    rowidx = torch.randint(0, 1000, (num_edges,)).to(torch.int32)
    colidx = torch.randint(0, 1000, (num_edges,)).to(torch.int32)

    d = run.cpu.compile_and_run(code, edge_idx, rowidx, colidx)

    res = torch.zeros_like(edge_idx)
    for i in range(len(edge_idx)):
        e = edge_idx[i]
        v0 = rowidx[e]
        v1 = colidx[e] + 1
        res[i] = v0 + v1

    print(d)
    print(res)
    print(torch.equal(d, res))






def apply_test2():
    d1 = Var('int')
    d2 = Var('int')
    A = Tensor((d1, d2))
    B = Tensor((d1, ))
    ast = A.apply(lambda x: x+B, axis=1)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)


def apply_test3():
    d1 = Var('int')
    d2 = Var('int')
    d3 = Var('int')
    A = Tensor((d1, d2), dtype='int')
    B = Tensor((d2, ), dtype='int')
    C = Tensor((d3, ), dtype='int')


    def apply_func(item):
        def apply_func2(item2):
            return C[item2] + B[item2]

        return item.apply(apply_func2)

    ast = A.apply(apply_func)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)



def apply_test4():
    d1 = Var('int')
    d2 = Var('int')
    d3 = Var('int')
    A = Tensor((d1, d2), dtype='int')
    B = Tensor((d2, ))


    def apply_func(item):
        def apply_func2(item2):
            return B[item2] + 1

        return item.apply(apply_func2)

    ast = A.apply(apply_func)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)

def apply_test5():
    A = Tensor((10, 20))
    B = Tensor((10, 20))
    res = apply(lambda a, b: a + b, (A, B))
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def apply_test6():
    A = Tensor((10, 20))
    B = Tensor((10, 20))
    res = apply(lambda a, b: a + b, (A, B), (1, 1))
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)



def apply_test7():
    A = Tensor((10, 20))
    B = Tensor((20, ))
    res = apply(lambda a, b: a + b, (A, B), (1, 0))
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


def apply_test8():
    A = Tensor((10, 20))
    ofs = A.apply(lambda a: a.size()).prefix_sum(inclusive=False)
    res = apply(lambda a: a + 1, (A, ), out_ofs=ofs)
    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)

def apply_test9():
    A = Tensor('A', (10, 20))
    ofs = A.apply(lambda a: a.size()).prefix_sum(inclusive=False)
    res = apply(lambda i, j: A[i:j] + 1, (ofs[:ofs._size()[0]-1], ofs[1:]), out_ofs=ofs)
    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)


def apply_test10():
    ndevs = 4
    nnodes = 1000
    dim = 512
    batch_size = 32
    buffer_size = nnodes // ndevs
    buffers = Tensor('buffers', (ndevs, buffer_size, dim))
    dev_ids = Tensor('dev_ids', (nnodes, ), dtype='int')
    buf_ofs = Tensor('buf_ofs', (nnodes, ), dtype='int')

    sampled_nodes = Tensor('sampled_nodes', (batch_size, ), dtype='int')

    devs = Tensor('devs', (ndevs, ), dtype='int')

    # rowptr is obtained by sorting sampled_nodes according to their dev_ids
    rowptr = Tensor('rowptr', (ndevs+1, ), dtype='int')


    res = devs.apply(lambda d: buffers[rowptr[d]:rowptr[d+1]] + 1)
    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)


def apply_test11():
    A = Tensor((10, ))
    res = A[:5].apply(lambda x:x)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def cond_apply_test1():
    d1 = Var(name='d1')
    d2 = Var(name='d2')
    A = Tensor((d1, d2))
    B = Tensor((d1,))
    ast = A.apply(lambda x: x + 1, cond=B)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)


def cond_apply_test2():
    d1 = Var(name='d1')
    d2 = Var(name='d2')
    A = Tensor((d1, d2))
    B = Tensor((d2,))
    ast = A.apply(lambda x: x + 1, axis=1, cond=B)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)


def cond_apply_test3():
    d1 = Var(name='d1')
    d2 = Var(name='d2')
    A = Tensor((d1, d2))
    B = Tensor((d1,))
    C = Tensor((d1,))
    cond = bigger(B + C.abs(), 0)
    ast = A.apply(lambda x: x + 1, cond=cond)
    code = codegen.cpu.print_cpp(gen_ir(ast))
    print(code)


def view_apply_test1():
    A = Tensor((200, ))
    B = Tensor((10, 20))
    res = apply(lambda a, b: a + b, (A.view((10, 20), (0, 0)), B))
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def view_apply_test2():
    A = Tensor((200, ))
    B = Tensor((10, 20))
    res = apply(lambda a, b: a + b, (A, B.view((200, ), ([0, 1], ))))
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)



def view_apply_test3():
    A = Tensor((200, ))
    B = Tensor((20, ))
    res = apply(lambda a, b: a + b, (A.view((10, 20), (0, 0)), B), (1, 0))
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def view_apply_test4():
    A = Tensor((200, ))
    B = Tensor((10, 20))
    res = apply(lambda a, b: a + b, (A.view((4, 10, 20), (-1, 0, 0)), B.view((4, 10, 20), (-1, 0, 1))))
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


def test_aggr1():
    A = Tensor('A', (10, 20))
    indices = Tensor('idx', (A._size()[0], ), dtype='int')
    res = A.aggr_max(indices)

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)


def spmv():
    m = Var('m', 'int')
    r = Var('r', 'int')
    rowidx = Tensor('ridx', (m, ), 'int')
    colidx = Tensor('cidx', (m, ), 'int')
    val = Tensor('val', (m, ), 'float')

    c = Var('c', 'int')
    y = Tensor('y', (c, 100), 'float')

    res = y[colidx] * val
    res = res.aggr_sum(rowidx, size=r)

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)

def test_matmul():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    A = Tensor('a', (d1, d2))
    B = Tensor('b', (d2, d3))
    C = A @ B
    ir = gen_ir(C)
    code = codegen.cpu.print_cpp(ir)
    print(helpers.get_input_nodes(ir))


    d1 = 100
    d3 = 30
    d2 = 20
    A = torch.rand(d1, d2)
    B = torch.rand(d2, d3)
    res = run.cpu.compile_and_run(code, d1, d3, d2, A, B)
    print(torch.equal(A @ B, res))


def test_einsum1():
    d1 = Var('d1')
    d2 = Var('d2')
    A = Tensor('a', (d1, ))
    B = Tensor('b', (d2, ))
    C = einsum('i,j->ij', A, B)
    ir = gen_ir(C)
    code = codegen.cpu.print_cpp(ir)
    print(helpers.get_input_nodes(ir))
    print(code)


    d1 = 100
    d3 = 30
    d2 = 20
    A = torch.rand(d1, )
    B = torch.rand(d2, )
    res = run.cpu.compile_and_run(code, d1, d2, A, B)
    print(torch.equal(torch.einsum('i,j->ij', A, B), res))


def test_einsum2():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    d4 = Var('d4')
    A = Tensor('a', (d1, d2, d3))
    B = Tensor('b', (d1, d3, d4))
    C = einsum('bij,bjk->ik', A, B)
    ir = gen_ir(C)
    code = codegen.cpu.print_cpp(ir)
    print(helpers.get_input_nodes(ir))
    print(code)


    d1 = 8
    d2 = 20
    d3 = 30
    d4 = 10

    A = torch.rand(d1, d2, d3)
    B = torch.rand(d1, d3, d4)
    res = run.cpu.compile_and_run(code, d2, d4, d1, d3, A, B)
    # TODO: cuke computes differently than pytorch when 'ik' changes to 'ki', it seems pytorch does not differentiate the two
    res1 = torch.einsum('bij,bjk->ik', A, B)
    print(torch.norm(res1 - res))


def einsum_test3():
    d = 20
    A = Tensor('a', (d, ))
    B = Tensor('b', (d, ))
    C = einsum('i,i->', A, B)
    ir = gen_ir(C)
    code = codegen.cpu.print_cpp(ir)
    print(helpers.get_input_nodes(ir))
    print(code)


    A = torch.rand(d, )
    B = torch.rand(d, )
    res = run.cpu.compile_and_run(code, A, B)
    print(torch.einsum('i,i->', A, B), res)


def test_apply5():
    m = Var('nedges')
    n = Var('nnodes')
    rowidx = Tensor('rowidx', (m, ), dtype='int')
    colidx = Tensor('colidx', (m, ), dtype='int')
    rowptr = Tensor('rowptr', (n+1, ), dtype='int')

    v0 = rowidx[0]
    v1 = colidx[1]
    v0_nb = colidx[rowptr[v0]: rowptr[v0+1]].sum() +  colidx[rowptr[v1]: rowptr[v1+1]].sum()

    code = codegen.cpu.print_cpp(v0_nb._gen_ir())
    print(code)

def test22():
    d = Var('d')
    c = Var('c')
    D = Tensor('D', (10, ), dtype='int')
    A = Tensor('A', (d, d+d, D[c]*d))
    B = Tensor('B', (d, d+c, c*d))
    b1 = Var('b1')
    b2 = Var('b2')
    idx = Tensor('idx', (5, ), dtype='int')

    ast = A[1:3][idx][b1:b2] + B[1:3][idx][b1:b2]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    d = 7
    c = 6
    D = torch.randint(10, 15, (10, ), dtype=torch.int)
    A = torch.rand(d, d+d, D[c]*d)
    B = torch.rand(d, d+c, c*d)
    b1 = 4
    b2 = 5
    idx = torch.IntTensor(5)

    d = run.cpu.compile_and_run(code, b2, b1, D, c, d, A, idx, B)

    print(A[1:3][:,idx][:,:,b1:b2] + B[1:3][:,idx][:,:,b1:b2])
    print(torch.equal(A[1:3][:,idx][:,:,b1:b2] + B[1:3][:,idx][:,:,b1:b2], d))

def test23():
    d1 = Var('d1')
    d2 = Var('d2')
    A = Tensor('A', (d1, d2))
    B = Tensor('B', (d2, ))
    ast = A.apply(lambda x: x+B, axis=0)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))

    code = codegen.cpu.print_cpp(ir)

    d1 = 10
    d2 = 20
    A = torch.ones(d1, d2)
    B = torch.ones(d2)
    res = run.cpu.compile_and_run(code, d1, d2, A, B)
    print(res)



def test24():
    nnodes = Var('nnodes')
    max_d = Var('max_d')
    G = Tensor('G', (nnodes, max_d), dtype='int')
    V = Tensor('V', (nnodes, ), dtype='int')

    def backtrack1(v0):
        def backtrack2(v1):
            return (G[v1] + G[v0]).size()

        res = G[v0].apply(backtrack2)
        res = res.sum()

        return res

    res = V.apply(backtrack1)
    res = res.sum()
    ir = gen_ir(res)
    print(helpers.get_input_nodes(ir))

    code = codegen.cpu.print_cpp(ir)

    nnodes = 100
    max_d = 50
    G = torch.randint(0,100, (nnodes, max_d), dtype=torch.int)
    V = torch.arange(0, nnodes, dtype=torch.int)

    res = run.cpu.compile_and_run(code, nnodes,V, max_d, G)
    print(res)





def test25():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    A = Tensor('A', (d1, d2, d3))
    B = Tensor('B', (d1, d2))
    ast = A.apply(lambda x: x+B, axis=2)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))

    code = codegen.cpu.print_cpp(ir)

    d1 = 10
    d2 = 20
    d3 = 30
    A = torch.ones(d1, d2, d3)
    B = torch.ones(d1, d2)
    res = run.cpu.compile_and_run(code, d3, d1, d2, A, B)
    print(res)


def test26():
    A = Tensor('a', (10, ))
    ast = A.sum(axis=0)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)

    A = torch.rand(10)
    init = 0.0
    res = run.cpu.compile_and_run(code, A)
    print(res, torch.sum(A))

def reduce_test1():
    A = Tensor((10, 20))
    res = A.sum(axis=0)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.sum(A, dim=0))


def reduce_test2():
    A = Tensor((10, 20))
    res = A.sum(axis=1)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.sum(A, dim=1))

def reduce_test3():
    A = Tensor((10, 20, 5))
    res = A.sum(axis=1)
    code = codegen.cpu.print_cpp(gen_ir(res))

    print(code)

    A = torch.rand(10, 20, 5)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.sum(A, dim=1))


def reduce_test4():
    A = Tensor((10, 20, 5))
    res = A.max(axis=0)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20, 5)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.max(A, dim=0).values)


def reduce_test5():
    A = Tensor((10, 20, 5))
    res = A.max(axis=1)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20, 5)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.max(A, dim=1).values)

def reduce_test6():
    A = Tensor((10, 20, 5))
    res = A.max(axis=2)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20, 5)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.max(A, dim=2).values)

def reduce_test7():
    A = Tensor((10, 20))
    res = A.min(axis=0)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, 20)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.min(A, dim=0).values)

def reduce_test8():
    A = Tensor((10, ))
    res = A.min(axis=0)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, )
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.min(A, dim=0).values)

def reduce_test9():
    A = Tensor((10, ))
    res = A.max(axis=0)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

    A = torch.rand(10, )
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.max(A, dim=0).values)

def test29():
    A = Tensor('A', (10, ))
    B = Tensor('B', (10, ), dtype='int')
    res = A[B[0]] + A[B[1]]

    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


def conv1d_v1():
    A = Tensor('a', (100, ))
    ast = A[0:97] + A[1:98] + A[2:99]
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)
    print(code)

def conv1d_v2(width):
    A = Tensor('a', (100, ))
    res = Tensor('t', A[width:]._size()).setval(0)
    for i in range(width):
        res = res + A[i:i+97]
    ir = gen_ir(res)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)
    print(code)

def cmp_test():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (10, 20))
    res = bigger(A, B)

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)


def scan_test1():
    A = Tensor('a', (10, ))
    res = A.prefix_sum()

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)

def scan_test2():
    A = Tensor('a', (10, 20))
    res = A.prefix_sum()

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)

def redirect_test():
    data = Tensor('data', (10, 20), dtype='float')
    res = (data + 1) >> data
    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)

def prefix_sum1():
    data = Tensor('data', (10, ), dtype='float')
    psum = data.prefix_sum()
    code = codegen.cpu.print_cpp(psum._gen_ir())
    print(code)


def prefix_sum2():
    data = Tensor('data', (10, ), dtype='float')
    psum = data.prefix_sum(inclusive=False)
    code = codegen.cpu.print_cpp(psum._gen_ir())
    print(code)

def prefix_sum3():
    data = Tensor('data', (10, 30), dtype='float')
    psum = data.prefix_sum(axis=0, inclusive=False)
    code = codegen.cpu.print_cpp(psum._gen_ir())
    print(code)

def prefix_sum4():
    data = Tensor('data', (10, 30), dtype='float')
    psum = data.prefix_sum(axis=1)
    code = codegen.cpu.print_cpp(psum._gen_ir())
    print(code)


def view_test1():
    data = Tensor((1024, ))
    view1 = data.view((16, 64, 64), (0, -1, 0))
    view2 = view1.view((1024, 64), ([0,1], 2))

    gen_ir(view2)
    # print([s.val for s in view1.ref_size])
    # print(view1.eval.size[0].val)
    # print(view1.attr['dim_map'])
    # print([s.val for s in view1.attr['size_map']])
    # print('--------------------')
    # print([s.val for s in view2.ref_size])
    # print(view2.eval.size[0].val)
    # print(view2.attr['dim_map'])
    # print([s.val for s in view2.attr['size_map'][0]])
    # print(view2.attr['size_map'][1].val)

def view_test2():
    data = Tensor((1024, ))
    view1 = data.view((16, 64), (0, 0))
    i = Var(name='i')

    tmp1 = view1[i]

    view2 = tmp1.view((4, 16), (0, 0))

    k = Var(name='k')
    tmp2 = view2[k]
    tmp3 = Tensor(tmp2.ref_size)
    res = tmp2 + tmp3

    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)

    print(view2.attr['dim_map'])
    print([s.val for s in view2.attr['size_map']])


def view_test3():
    data = Tensor((1024, ))
    view1 = data.view((16, 64), (0, 0))
    i = Var(name='i')

    tmp1 = view1[:, i]

    view2 = tmp1.view((4, 4), (0, 0))

    k = Var(name='k')
    tmp2 = view2[k]
    tmp3 = Tensor(tmp2.ref_size)
    res = tmp2 + tmp3

    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)

    # print(tmp1.eval)
    print(view2.attr['dim_map'])
    print([s.val for s in view2.attr['size_map']])

def view_test4():
    data = Tensor((1024, ))
    view1 = data.view((16, 64), (0, 0))
    i = Var(name='i')

    tmp1 = view1[:, i]

    view2 = tmp1.view((4, 4), (0, 0))

    k = Var(name='k')
    tmp2 = view2[:, k]
    tmp3 = Tensor(tmp2.ref_size)
    res = tmp2 + tmp3

    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)

    print(tmp1.eval)
    print(view2.attr['dim_map'])
    print([s.val for s in view2.attr['size_map']])

def view_test5():
    data = Tensor((1024, ))
    view1 = data.view((16, 8, 8), (0, 0, 0))
    i = Var(name='i')

    tmp1 = view1[i]
    tmp2 = Tensor(tmp1.ref_size)
    res = tmp1 + tmp2
    ir = gen_ir(res)

    code = codegen.cpu.print_cpp(ir)
    print(code)


def view_test6():
    data = Tensor((1024, ))
    view1 = data.view((16, 8, 8), (0, 0, 0))
    i = Var(name='i')
    j = Var(name='j')

    tmp1 = view1[:, i, j]
    tmp2 = Tensor(tmp1.ref_size)
    res = tmp1 + tmp2
    ir = gen_ir(res)

    code = codegen.cpu.print_cpp(ir)
    print(code)


def view_test7():
    data = Tensor((1024, ))
    view1 = data.view((16, 64, 64), (0, -1, 0))
    tmp1 = Tensor(view1.ref_size)
    res = view1 + tmp1

    ir = gen_ir(res)

    print(view1.ref_size)
    print(view1.eval.size)
    print(view1.attr['dim_map'])
    print([s.val for s in view1.attr['size_map']])
    code = codegen.cpu.print_cpp(ir)
    print(code)


def view_test8():
    data = Tensor((1024, ))
    view1 = data.view((16, 64, 64), (0, -1, 0))
    i = Var(name='i')
    tmp1 = view1[i]
    tmp2 = Tensor(tmp1.ref_size)
    res = tmp1 + tmp2

    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)

    print(tmp1.attr['dim_map'])
    print(tmp1.eval.size)

def view_test9():
    data = Tensor((4, 8))
    view1 = data.view((32, ), ([0, 1], ))
    i = Var(name='i')
    tmp1 = view1[i]
    tmp2 = Var(name='tmp2')
    res = tmp1 + tmp2
    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)
    print(tmp1.ref_size)
    print(tmp1.eval.size)
    print(tmp1.attr['dim_map'])
    print(tmp1.attr['size_map'])


def view_test10():
    data = Tensor((4, 8))
    view1 = data.view((32, 16), ([0, 1], -1))
    i = Var(name='i')
    tmp1 = view1[i]
    tmp2 = Tensor(tmp1.ref_size)
    res = tmp1 + tmp2
    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)
    print(tmp1.ref_size)
    print(tmp1.eval.size)
    print(tmp1.attr['dim_map'])
    print(tmp1.attr['size_map'])


def view_test11():
    data = Tensor((16, 32))
    view1 = data.view((2, 8, 8, 4), (0, 0, 1, 1))
    i = Var(name='i')
    j = Var(name='j')
    tmp1 = view1[i, :, :, j]
    tmp2 = Tensor(tmp1.ref_size)
    res = tmp1 + tmp2
    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)



def neg_transE():
    class fuser:
        def __init__(self):
            self.rules = [transform.fuse.basic_rule]

        def __call__(self, node):
            def action(n, res):
                for r in self.rules:
                    r(n, res)

            t = helpers.ASGTraversal(action)
            t(node)
            return node


    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batchsize = Var(name='batchsize')
    blocksize = Var(name='blocksize')

    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    h = Tensor((batchsize, ), dtype='int', name='h')
    t = Tensor((batchsize, ), dtype='int', name='t')
    r = Tensor((batchsize, ), dtype='int', name='r')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    nblocks = setval(batchsize / blocksize)

    vt_view1 = vt.view((nblocks, blocksize, blocksize, dim), (0, -1, 0, 1))
    vt_view2 = vt_view1.view((batchsize, blocksize, dim), ([0, 1], 2, 3))
    vh = vh.view((batchsize, blocksize, dim), (0, -1, 1))
    vr = vr.view((batchsize, blocksize, dim), (0, -1, 1))

    res = vh - vt_view2 + vr
    transform.passes = [fuser()]
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


def neg_transR():
    def bvm(a, b):
        return apply(lambda x, y: einsum('i,ij->j', x, y), (a, b))

    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batchsize = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    h = Tensor((batchsize, ), dtype='int', name='h')
    t = Tensor((batchsize, ), dtype='int', name='t')
    r = Tensor((batchsize, ), dtype='int', name='r')

    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]
    vr = Remb[r]

    blocksize = Var(name='blocksize')
    nblocks = setval(batchsize / blocksize, name='nblocks')

    output_size = setval(batchsize * blocksize, name='output_size')

    vt = vt.view((nblocks, blocksize, blocksize, dim), (0, -1, 0, 1)).view((batchsize, blocksize, dim), ([0, 1], 2, 3)).view((output_size, dim), ([0, 1], 2))
    vh = vh.view((batchsize, blocksize, dim), (0, -1, 1)).view((output_size, dim), ([0, 1], 2))
    vr = vr.view((batchsize, blocksize, dim), (0, -1, 1)).view((output_size, dim), ([0, 1], 2))
    mr = mr.view((batchsize, blocksize, dim, dim), (0, -1, 1, 2))


    res = bvm((vh - vt), mr.view((output_size, dim, dim), ([0, 1], 2, 3))) + vr
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


def var_test1():
    x = Var(name='x', dtype='float')
    y = x / 10 + 2
    res = Var(name='res')
    res = setval(y)
    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)


def var_test2():
    x1 = Var(dtype='float', name='x')
    x2 = Var(dtype='float', name='x')
    y = x1.abs() / 10 + 2 * x2.abs()
    res = Var(name='res')
    res = setval(y.round())
    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)


def psum_test1():
    data = Tensor((10, 20))
    res = data.prefix_sum(axis=0, inclusive=True)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def psum_test2():
    data = Tensor((10, 20))
    res = data.prefix_sum(axis=0, inclusive=False)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def psum_test3():
    data = Tensor((10, 20))
    res = data.prefix_sum(axis=1, inclusive=True)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

def psum_test4():
    data = Tensor((10, 20))
    res = data.prefix_sum(axis=1, inclusive=False)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)

if __name__ == "__main__":
    # basic tensor indexing tests
    # test1() # pass
    # test2() # pass
    # test3() # pass
    # test4() # pass
    # test5() # pass
    # test6() # pass
    # test7() # pass
    # test8() # pass
    # test9() # pass
    # test9_1() # pass
    # test10() # pass
    # test10_1() # pass
    # test10_2() # pass
    # test10_3() # pass
    # test10_4() # pass
    # test10_5() # pass
    # test11()
    # test12()
    # test13()
    # test13_1()
    # test14()
    # some slicing examples
    # test15()
    # test16()
    # test17()
    # test18()
    # test18_1()
    # test19()
    # test20()
    # test21()
    # compression()
    # test_math1()
    # test_math2()
    apply_test1() # pass
    apply_test2() # pass
    apply_test3() # pass
    apply_test4() # pass
    apply_test5() # pass
    apply_test6() # pass
    apply_test7()  # pass
    # apply_test8()
    # apply_test9()
    # apply_test10()
    # apply_test11() # pass
    # cond_apply_test1() # pass
    # cond_apply_test2() # pass
    # cond_apply_test3() # pass

    # view_apply_test1() # pass
    # view_apply_test2() # pass
    # view_apply_test3() # pass
    # view_apply_test4() # pass


    reduce_test1()  # pass
    reduce_test2()  # pass
    reduce_test3() # pass
    # reduce_test4() # pass
    # reduce_test5() # pass
    # reduce_test6() # pass
    # reduce_test7() # pass
    # reduce_test8() # pass
    # reduce_test9() # pass
    # test_aggr1()
    # spmv()
    # test_einsum1()
    # test_einsum2()
    # einsum_test3()
    # apply_test2()
    # test_apply5()
    # test27()
    # scan_test1()
    # scan_test2()
    # cmp_test()
    # prefix_sum1()
    # prefix_sum2()
    # prefix_sum3()
    # prefix_sum4()
    # view tests
    # view_test1()
    # view_test2()
    # view_test3()
    # view_test4()
    # view_test5()
    # view_test6()
    # view_test7()
    # view_test8()
    # view_test9()
    # view_test10()
    # view_test11()
    # neg_transE()
    # neg_transR()


    # var_test1() # pass
    # var_test2() # pass

    # psum_test1()
    # psum_test2()
    # psum_test3()
    # psum_test4()
