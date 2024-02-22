import transform
from codegen import *
from helpers import new_op, ASGTraversal
from transform.fuse import basic_rule, fuse_operators
from asg import *
from asg2ir import gen_ir

@new_op
def bvv(a, b):
    return apply(lambda x, y: einsum('i,i->', x, y), (a, b))

@new_op
def bsv(a, b):
    return apply(lambda x, y: x * y, (a, b))

@new_op
def bvm(a, b):
    return apply(lambda x, y: einsum('i,ij->j', x, y), (a, b))

@new_op
def bov(a, b):
    return apply(lambda x, y: einsum('i,j->ij', x, y), (a, b))


def fuse_rule(node, res):
    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bvv':
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op or ('op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] == 'bvm'):
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op or ('op_name' in node.operators[2].attr and node.operators[2].attr['op_name'] == 'bvm'):
                fuse_operators(node, node.input_orders[2], node.operators[2])

    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bsv':
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if 'op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] == 'bvv':
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[2], node.operators[2])

    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bov':
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[2], node.operators[2])


class fuser:
    def __init__(self):
        self.rules = [basic_rule, fuse_rule]

    def __call__(self, node):
        def action(n, res):
            for r in self.rules:
                r(n, res)
        t = ASGTraversal(action)
        t(node)
        return node

class tiler():

    def __init__(self, C, D):
        self.C = C
        self.D = D

    def __call__(self, node):
        def action(n, res):
            if not 'scope' in n.attr and len(n.compute) > 0:
                transform.split.split_level(n, self.C, 0)
                transform.parallelize.parallelize_loop(n, 80, [0])
                transform.parallelize.parallelize_loop(n, 16, [0, 0])
                transform.split.split_level(n, self.D, 2)
                transform.parallelize.parallelize_level(n, 32, 3)
                if 'op_name' in n.attr and n.attr['op_name'] == 'bov':
                    transform.split.split_axis(n, self.D, 2)

        t = ASGTraversal(action)
        t(node)
        return node

class smem():
    def __init__(self):
        pass

    def __call__(self, node):
        def action(n, res):
            if type(n) == TensorOp:
                if 'op_name' in n.attr and n.attr['op_name'] in ('bvv', 'bsv', 'bmv'):
                    transform.cuda_smem.add_direct_cache(n.eval, node) # this function should 1) create a shared memory tensor for n.eval, 2) change all reference to the shared memory tensor for code in the scope of node, 3) handle both reduction ore non-reduction data
                if n.op_type == 'index' and 'reuse' in n.operators[1][0].attr and n.operators[1][0].attr['reuse'] == True:
                    transform.cuda_smem.add_indirect_cache(n.eval, node) # this function should 1) create a shared memory tensor for n.eval, 2) analyze n.eval.idx to get buf_idx/uniq_idx, 3) change all reference based on buf_idx/uniq_idx for code in the scope of node

        self.eval = node.eval
        t = ASGTraversal(action)
        t(node)
        
        return node

# transform.passes = [f, tiler(16, 128), parallelizer([80, 8, 32])]
# transform.passes = [f]
# transform.passes = [fuser(), tiler(16, 128)]
transform.passes = [fuser()]
# transform.passes = [fuser(), tiler(16, 64)]
# transform.passes = [fuser(), tiler(16, 64), smem(16, 64)]


def transE():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb', )
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = vh - vt + vr
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)


def transH():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Pemb = Tensor((nedges, dim), name='Pemb')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    vp = Pemb[r]

    # TODO: if there are redundant computation, is fusion always beneficial
    res = vh - vt + vr - bsv(bvv(vp, vh - vt), vp)
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)


def transR():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]
    vr = Remb[r]

    res = bvm(vh - vt, mr) + vr
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)


def transF():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = bvv(vh, vt) - bvv(vh - vt, vr)
    code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    # code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)


def RESCAL():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]

    res = bvv(bvm(vh, mr), vt)
    code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    # code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)



def backward():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = bov(vh+vr, vt-vr)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


if __name__ == "__main__":
    # transE()
    # transH()
    # transR()
    # transF()
    RESCAL()
    # backward()