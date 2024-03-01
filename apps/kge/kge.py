import transform
import run
from codegen import *
from helpers import new_op, ASGTraversal, IRTraversal, flatten, get_obj
from transform.fuse import basic_rule, fuse_operators
from asg import *
from asg2ir import gen_ir
from ir import *

import torch
from torch.utils.cpp_extension import load

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
        self.flag = []

    def __call__(self, node):
        def action(n, res):
            if isinstance(n, TensorOp) and 'op_name' in n.attr:
                self.flag.append(n.attr['op_name'])
            if not 'scope' in n.attr and len(n.compute) > 0:
                transform.split.split_level(n, self.C, 0)
                transform.parallelize.parallelize_loop(n, 80, [0])
                transform.parallelize.parallelize_loop(n, 16, [0, 0])
                transform.split.split_level(n, self.D, 2)
                transform.parallelize.parallelize_level(n, 64, 3)
                if 'bvm' in self.flag:
                    transform.split.split_level(n, self.D, 4)
                if 'op_name' in n.attr and n.attr['op_name'] == 'bov':
                    transform.split.split_axis(n, self.D, 2)

        t = ASGTraversal(action)
        t(node)
        return node

class smem():
    def __init__(self, C, D):
        self.C = C
        self.D = D

    def __call__(self, node):
        def action(n, res):
            if type(n) == TensorOp and 'op_name' in n.attr and n.attr['op_name'] in ['bsv', 'bvv', 'bvm']:
                
                transform.cuda_smem.add_direct_cache(node, n.eval)
                # this function should 1) create a shared memory tensor for n.eval, 2) change all reference to the shared memory tensor for code in the scope of node, 3) handle both reduction ore non-reduction data
            
            if type(n) == TensorOp and n.op_type == 'index' and 'reuse' in n.operators[1].attr and n.operators[1].attr['reuse'] == True:
                unique_idx = Tensor((n.operators[1]._size()[0]/self.C, self.C), dtype='int', name=n.operators[1].name+'_uniq')
                buf_idx = Tensor((n.operators[1]._size()[0]/self.C, self.C), dtype='int', name=n.operators[1].name+'_buf')
                unique_cnt = Tensor((n.operators[1]._size()[0]/self.C, ), dtype='int', name=n.operators[1].name+'_unique_cnt')
                n.operators[1].attr['idx'] = [[unique_idx.name, unique_idx], [buf_idx.name, buf_idx], [unique_cnt.name, unique_cnt]]
                
                # transform.cuda_smem.add_indirect_cache(node, n.eval, self.C, self.D)
                # this function should 1) create a shared memory tensor for n.eval, 2) analyze n.eval.idx to get buf_idx/uniq_idx, 3) change all reference based on buf_idx/uniq_idx for code in the scope of node
                # traverse n.eval.attr['cache'] array or scalar
                # tensor should be generated in add_ func, gen_ir()
                transform.cuda_smem.add_indirect_cache(node, n.eval, self.C, self.D, unique_idx, buf_idx, unique_cnt)
            
            if type(n) == TensorOp and 'op_name' in n.attr and n.attr['op_name'] in ['bvv', 'bvm']:
                def get_assigns(s, res):
                    if type(s) in (Assignment, Code):
                        res.append(s)
                    return [True, True, True, True, True]
                
                def _find_reduction_loop(loop):
                    def action(s, res):
                        if isinstance(s, Loop):
                            if 'ptype' in s.attr and s.attr['ptype'] == 'reduction':
                                res.append(s)
                        return [True, True, True, True, True]

                    r = IRTraversal(action)(loop)
                    return r
                scope = flatten(node.compute)
                for loop in scope:
                    redu_loop = _find_reduction_loop(loop)
                    for s in redu_loop:
                        assign = IRTraversal(get_assigns)(s)
                        # get lhs of last assignment
                        lhs = assign[-1].lhs
                        obj = get_obj(lhs)
                        transform.cuda_smem.add_direct_cache(node, obj)
                

        self.eval = node.eval
        t = ASGTraversal(action)(node)
        return node

# transform.passes = [f, tiler(16, 128), parallelizer([80, 8, 32])]
# transform.passes = [f]
# transform.passes = [fuser(), tiler(16, 128)]
# transform.passes = [fuser()]
# transform.passes = [fuser(), tiler(16, 64)]
transform.passes = [fuser(), tiler(16, 64), smem(16, 64)]


def inspector(h, t, r, rel_num):
    C = 16
    # print(h.shape)
    uniq = torch.zeros((h.shape[0]//C, C), dtype=torch.int64).cuda(0)
    buf = torch.zeros((h.shape[0]//C, C), dtype=torch.int64).cuda(0)
    cnt = torch.zeros((h.shape[0]//C, ), dtype=torch.int64).cuda(0)

    module = load(name='sort', sources=['apps/kge/inspection/inspector.cu'])
    x = module.gpu_sort(h, t, r, uniq, buf, cnt, int(h.shape[0]), int(rel_num))
    # print(x)
    # print(uniq, buf, cnt, r)
    return uniq, buf, cnt

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
    r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = vh - vt + vr
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    batchsize=1024
    dimension=512
    relations = 30
    entities = 999
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)

    # y = eemb[hh] - eemb[tt] + remb[rr]
    # print(y, y.shape)

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr)
    # print(x, x.shape)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))

    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]

    # uniq, buf, cnt = inspector(hh, tt, rr, relations)
    # # print(uniq, buf, cnt)

    # # cnt = torch.zeros_like(cnt).cuda(0)
    # y = eemb[hh] - eemb[tt] + remb[rr]
    # print(y, y.shape)
    # # print(eemb.dtype, hh.dtype, remb.dtype, rr.dtype, uniq.dtype, buf.dtype, cnt.dtype, tt.dtype)
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, uniq, buf, cnt)
    # print(x, x.shape)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))

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

    batchsize=512
    dimension=512
    relations = 51
    entities = 9999
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension)).cuda(0)

    # for not reuse test
    # y = eemb[hh] - eemb[tt] + remb[rr] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', pemb[rr], eemb[hh]-eemb[tt]), pemb[rr])
    # print(y, y.shape)

    # print(torch.einsum('ab,ab->a', pemb[rr], eemb[hh]-eemb[tt])[:16])

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, pemb)
    # print(x, x.shape)

    # reuse index building test
    indices = torch.argsort(rr)
    hh = hh[indices]
    tt = tt[indices]
    rr = rr[indices]

    uniq, buf, cnt = inspector(hh, tt, rr, relations)
    print(uniq, buf, cnt)

    y = y = eemb[hh] - eemb[tt] + remb[rr] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', pemb[rr], eemb[hh]-eemb[tt]), pemb[rr])
    print(y, y.shape)

    x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, uniq, buf, cnt, pemb)
    print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)), torch.sum(x - y))


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

    batchsize=512
    dimension=256
    relations = 30
    entities = 9999
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension, dimension)).cuda(0)

    # y = torch.einsum('ab,abc->ac', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr]
    # print(y, y.shape)

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, remb)
    # print(x, x.shape)

    indices = torch.argsort(rr)
    hh = hh[indices]
    tt = tt[indices]
    rr = rr[indices]

    uniq, buf, cnt = inspector(hh, tt, rr, relations)
    print(uniq, buf, cnt)

    y = torch.einsum('ab,abc->ac', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr]
    print(y, y.shape)
    # print(eemb.dtype, hh.dtype, remb.dtype, rr.dtype, uniq.dtype, buf.dtype, cnt.dtype, tt.dtype)
    x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, uniq, buf, cnt, remb)
    print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)))


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
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    batchsize=512
    dimension=512
    relations = 30
    entities = 9999
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)

    # y = torch.einsum('ab,ab->a', eemb[hh], eemb[tt]) - torch.einsum('ab,ab->a',(eemb[hh] - eemb[tt]), remb[rr])
    # print(y, y.shape)

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr)
    # print(x, x.shape)

    indices = torch.argsort(rr)
    hh = hh[indices]
    tt = tt[indices]
    rr = rr[indices]

    uniq, buf, cnt = inspector(hh, tt, rr, relations)
    print(rr, uniq, buf, cnt)
    y = torch.einsum('ab,ab->a', eemb[hh], eemb[tt]) - torch.einsum('ab,ab->a',(eemb[hh] - eemb[tt]), remb[rr])
    print(y, y.shape)

    x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, uniq, buf, cnt)
    print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)))


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
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    batchsize=512
    dimension=512
    relations = 30
    entities = 999
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension, dimension)).cuda(0)

    # y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[hh], remb[rr]), eemb[tt])
    # print(y, y.shape)

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, 0, remb, rr, tt)
    # print(x, x.shape)

    indices = torch.argsort(rr)
    hh = hh[indices]
    tt = tt[indices]
    rr = rr[indices]

    uniq, buf, cnt = inspector(hh, tt, rr, relations)
    # print(uniq, buf, cnt)

    y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[hh], remb[rr]), eemb[tt])
    print(y, y.shape)
    # print(eemb.dtype, hh.dtype, remb.dtype, rr.dtype, uniq.dtype, buf.dtype, cnt.dtype, tt.dtype)
    x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, 0, remb, rr, uniq, buf, cnt, tt)
    print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)))



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