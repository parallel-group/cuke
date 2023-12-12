import codegen.cpu
import transform
from asg import *
from asg2ir import gen_ir
from transform import parallelize


def is_in(x, li):
    src = inspect.cleandoc("""
    for (int i=0; i<LSIZE; i++) {
        F = true;
    }
    """)
    found = Var(dtype='int')
    found.attr['is_arg'] = False
    return inline(src, ('F', found), ('X', x), ('LI', li), ('LSIZE', li._size()[0]))

def intersect(a, b):
    c = a.apply(lambda x: is_in(x, b))
    return a.apply(lambda x: x, cond=c)


class Graph:
    def __init__(self, rowptr, colidx):
        self.rowptr = rowptr
        self.colidx = colidx

    def get_neighbor(self, v):
        return self.colidx[self.rowptr[v]:self.rowptr[v + 1]]


def test0():
    A = Tensor((10, ))
    B = Tensor((10, ))
    res = A.apply(lambda x:x, cond=B)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


def test1():
    A = Tensor((10, ))
    B = Tensor((20, ))
    res = intersect(A, B)
    ir = gen_ir(res)
    code = codegen.cpu.print_cpp(ir)
    print(code)



def test2():
    A = Tensor((10, ))
    B = Tensor((20, ))
    C = Tensor((30, ))
    res = intersect(intersect(A, B), C)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)



def test3():
    nnodes = 100
    nedges = 1000
    edges = Tensor((nedges, 2), dtype='int', name='edges')
    rowptr = Tensor((nnodes + 1, ), dtype='int', name='rowptr')
    colidx = Tensor((nedges, ), dtype='int', name='colidx')
    g = Graph(rowptr, colidx)
    res = edges.apply(lambda e: intersect(g.get_neighbor(e[0]), g.get_neighbor(e[1])).size(0))
    res = res.sum()
    res = gen_ir(res)
    transform.passes.append(parallelize.parallelizer())
    code = codegen.cpu.print_cpp(res)
    print(code)



if __name__ == "__main__":
    # test0()
    # test1()
    # test2()
    test3()