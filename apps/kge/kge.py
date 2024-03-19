import transform
import run
from codegen import *
from helpers import new_op, ASGTraversal, IRTraversal, flatten, get_obj, has_same_value
from transform.fuse import basic_rule, fuse_operators
from asg import *
from asg2ir import gen_ir
from ir import *

import torch
from torch.utils.cpp_extension import load

from loss import get_loss
from kge_score_func import * 

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

@new_op
def bsm(a, b):
    return apply(lambda x, y: einsum(',ij->ij', x, y), (a, b))

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
            if node.operators[1].op_type in elementwise_op or ('op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] == 'bvv'):
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[2], node.operators[2])
        
    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bsm':
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op or ('op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] == 'bvv'):
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op or ('op_name' in node.operators[2].attr and node.operators[2].attr['op_name'] == 'bov'):
                fuse_operators(node, node.input_orders[2], node.operators[2])

    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bov':
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[2], node.operators[2])
    if type(node) == TensorOp and node.op_type in elementwise_op:
        if 'op_name' in node.operators[0].attr and node.operators[0].attr['op_name'] in ['bvv'] and 'op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] in ['bvv']:
            def _find_dim_loops(s, res):
                if isinstance(s, Loop) and s.end.name() == 'dim':
                    res.append(s)
                return [True, True, True, True, True]
            # print(node.compute, codegen.gpu.to_string(node.compute))
            t = IRTraversal(_find_dim_loops)(node.compute)
            # print(t)
            # print(node.operators[0].compute)
            # print(codegen.gpu.to_string(t))
            # def action(s, res):
            #     if isinstance(s, Loop):
            #         pass
            #     return [True, True, True, True, True]
            # IRTraversal(action)(node.compute)

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
                    transform.interchange.interchange(n, [3, 4])
                # if 'op_name' in n.attr and n.attr['op_name'] == 'bov':
                #     print(codegen.gpu.to_string(n.compute))
                #     transform.split.split_axis(n, self.D, 2)

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
                
                if n.compute:
                    transform.cuda_smem.add_direct_cache(node, n.eval)
                else:
                    def action(nn, res):
                        transform.cuda_smem.add_direct_cache(nn, n.eval)
                        if nn.ref_by:
                            for i in nn.ref_by:
                                # if type(i) == TensorOp and 'op_name' in i.attr and i.attr['op_name'] in ['bsv', 'bvv', 'bvm']:
                                    transform.cuda_smem.add_direct_cache(i, n.eval)
                    ASGTraversal(action)(node)
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

            if type(n) == TensorOp and 'op_name' in n.attr and n.attr['op_name'] in ['bvv']:
                

                def recurrent_call(nn):
                    scope = flatten(nn.compute)
                    for loop in scope:
                        redu_loop = _find_reduction_loop(loop)
                        for s in redu_loop:
                            assign = IRTraversal(get_assigns)(s)
                            # get lhs of last assignment
                            lhs = assign[-1].lhs
                            obj = get_obj(lhs)
                            if obj in n.eval.attr['storage']:
                                transform.cuda_smem.add_direct_cache(nn, obj)
                    for i in nn.ref_by:
                        recurrent_call(i)
                recurrent_call(n)
            if type(n) == TensorOp and n.op_type == 'norm':
                transform.cuda_smem.add_direct_cache(node, n.eval)

                def recurrent_call(nn):
                    scope = flatten(nn.compute)
                    for loop in scope:
                        redu_loop = _find_reduction_loop(loop)
                        for s in redu_loop:
                            assign = IRTraversal(get_assigns)(s)
                            # get lhs of last assignment
                            lhs = assign[-1].lhs
                            obj = get_obj(lhs)
                            if obj in n.eval.attr['storage']:
                                transform.cuda_smem.add_direct_cache(nn, obj)
                    for i in nn.ref_by:
                        recurrent_call(i)
                recurrent_call(n)
                

        self.eval = node.eval
        t = ASGTraversal(action)(node)
        return node

# transform.passes = [f, tiler(16, 128), parallelizer([80, 8, 32])]
# transform.passes = [fuser(), tiler(16, 128)]
# transform.passes = [fuser()]
# transform.passes = [fuser(), tiler(16, 64)]
transform.passes = [fuser(), tiler(16, 64), smem(16, 64)]


def inspector(r, rel_num):
    C = 16
    uniq = torch.zeros((r.shape[0]//C, C), dtype=torch.int64).cuda(0)
    buf = torch.zeros((r.shape[0]//C, C), dtype=torch.int64).cuda(0)
    cnt = torch.zeros((r.shape[0]//C, ), dtype=torch.int64).cuda(0)

    module = load(name='sort', sources=['apps/kge/inspection/inspector.cu'])
    module.gpu_sort(r, uniq, buf, cnt, int(r.shape[0]), int(rel_num))
    return uniq, buf, cnt

def transE():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    Pemb = Tensor((nedges, dim, dim), name='Pemb')
    mr = Pemb[r]

    # res = vh - vt + vr
    # # code = codegen.cpu.print_cpp(gen_ir(res))
    # # print(code)
    # code = codegen.gpu.print_cuda(gen_ir(res))
    # print(code)

    res = norm(vh - vt + vr, 2, dim)
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

    y = torch.norm(eemb[hh] - eemb[tt] + remb[rr], p=2, dim=-1)
    print(y, y.shape)

    x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr)
    print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)))

    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]

    # uniq, buf, cnt = inspector(hh, tt, rr, relations)
    # # print(uniq, buf, cnt)

    # # cnt = torch.zeros_like(cnt).cuda(0)
    # # y = eemb[hh] - eemb[tt] + remb[rr]
    # # print(y, y.shape)
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
    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]

    # uniq, buf, cnt = inspector(hh, tt, rr, relations)
    # # print(uniq, buf, cnt)

    # y = y = eemb[hh] - eemb[tt] + remb[rr] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', pemb[rr], eemb[hh]-eemb[tt]), pemb[rr])
    # # print(y, y.shape)

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, uniq, buf, cnt, pemb)
    # # print(x, x.shape)
    # print(torch.sum(torch.abs(x) - torch.abs(y)), torch.sum(x - y))


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

    # # y = torch.einsum('ab,abc->ac', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr]
    # # print(y, y.shape)

    # # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, remb)
    # # print(x, x.shape)

    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]

    # uniq, buf, cnt = inspector(hh, tt, rr, relations)
    # # print(uniq, buf, cnt)

    # y = torch.einsum('ab,abc->ac', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr]
    # print(y, y.shape)
    # # print(eemb.dtype, hh.dtype, remb.dtype, rr.dtype, uniq.dtype, buf.dtype, cnt.dtype, tt.dtype)
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, uniq, buf, cnt, remb)
    # print(x, x.shape)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))


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

    # res = bvv(vh, vt) - bvv(vh - vt, vr)
    res = bvv(vh+vr, vt) + bvv(vt-vr, vh)
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    # batchsize=512
    # dimension=512
    # relations = 30
    # entities = 9999
    # hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    # rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    # tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    # eemb = torch.rand((entities, dimension)).cuda(0)
    # remb = torch.rand((relations, dimension)).cuda(0)

    # y = torch.einsum('ab,ab->a', eemb[hh], eemb[tt]) - torch.einsum('ab,ab->a',(eemb[hh] - eemb[tt]), remb[rr])
    # print(y, y.shape)

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr)
    # print(x, x.shape)

    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]

    # uniq, buf, cnt = inspector(hh, tt, rr, relations)
    # # print(rr, uniq, buf, cnt)
    # y = torch.einsum('ab,ab->a', eemb[hh], eemb[tt]) - torch.einsum('ab,ab->a',(eemb[hh] - eemb[tt]), remb[rr])
    # # print(y, y.shape)

    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, uniq, buf, cnt)
    # # print(x, x.shape)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))


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

    uniq, buf, cnt = inspector(rr, relations)
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
    # code = codegen.cpu.print_cpp(gen_ir(res))
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

def neg_TransE():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    neg_size = Var(name='neg_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb', )
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    # h.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = vh - vt + vr
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    batchsize=1024
    dimension=512
    relations = 30
    entities = 999
    negsize = 16
    hh = torch.randint(0, entities, (batchsize, )).unsqueeze(1).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).unsqueeze(1).cuda(0)
    tt = torch.randint(0, entities, (batchsize, negsize)).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)

    y = eemb[hh] - eemb[tt] + remb[rr]
    # print(y, y.shape)

    tt = tt.reshape(-1)
    hh = hh.repeat(1,negsize).reshape(-1)
    rr = rr.repeat(1,negsize).reshape(-1)
    
    x = run.gpu.compile_and_run(code, batchsize * negsize, dimension, 0, eemb, hh, tt, 0, remb, rr)
    x = x.reshape(-1, negsize, dimension)
    # print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)), torch.sum(x - y))

def neg_TransH():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    neg_size = Var(name='neg_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Pemb = Tensor((nedges, dim), name='Pemb')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
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
    negsize = 64
    hh = torch.randint(0, entities, (batchsize, )).unsqueeze(1).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).unsqueeze(1).cuda(0)
    tt = torch.randint(0, entities, (batchsize, negsize)).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension)).cuda(0)

    y = eemb[hh] - eemb[tt] + remb[rr] - torch.einsum('ad,acb->adb', torch.einsum('acb,adb->ad', pemb[rr], eemb[hh]-eemb[tt]), pemb[rr])
    print(y, y.shape)

    tt = tt.reshape(-1)
    hh = hh.repeat(1,negsize).reshape(-1)
    rr = rr.repeat(1,negsize).reshape(-1)

    x = run.gpu.compile_and_run(code, batchsize*negsize, dimension, 0, eemb, hh, tt, 0, remb, rr, pemb)
    x = x.reshape(-1, negsize, dimension)
    print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)), torch.sum(x-y))

def neg_TransF():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    neg_size = Var(name='neg_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    # res = negbvv(vh+vr, vt-zero) + negbvv(vh, vt-vr)
    res = bvv(vh+vr, vt) + bvv(vt-vr, vh)
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    batchsize=1024
    dimension=512
    relations = 30
    entities = 999
    negsize = 16
    hh = torch.randint(0, entities, (batchsize, )).unsqueeze(1).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).unsqueeze(1).cuda(0)
    tt = torch.randint(0, entities, (batchsize, negsize)).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)

    y1 = torch.einsum('acb,adb->ad', eemb[hh] + remb[rr], eemb[tt]) + torch.einsum('acb,adb->ad', (eemb[hh]), eemb[tt] - remb[rr])
    print(y1.shape)

    tt = tt.reshape(-1)
    hh = hh.repeat(1,negsize).reshape(-1)
    rr = rr.repeat(1,negsize).reshape(-1)

    y = torch.einsum('ab,ab->a', eemb[hh] + remb[rr], eemb[tt]) + torch.einsum('ab,ab->a',(eemb[hh]), eemb[tt] - remb[rr])
    y = y.reshape(-1, negsize)
    print(y.shape, torch.sum(torch.abs(y) - torch.abs(y1)))

    x = run.gpu.compile_and_run(code, batchsize*negsize, dimension, 0, eemb, hh, 0, remb, rr, tt)
    x = x.reshape(-1, negsize)
    # print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)), torch.sum(x - y))

def neg_TransR():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    neg_size = Var(name='neg_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
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
    negsize = 16
    hh = torch.randint(0, entities, (batchsize, )).unsqueeze(1).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).unsqueeze(1).cuda(0)
    tt = torch.randint(0, entities, (batchsize, negsize)).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension, dimension)).cuda(0)

    

    tt = tt.reshape(-1)
    hh = hh.repeat(1,negsize).reshape(-1)
    rr = rr.repeat(1,negsize).reshape(-1)

    y = torch.einsum('ab,abc->ac', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr]
    print(y, y.shape)

    x = run.gpu.compile_and_run(code, batchsize*negsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, remb)
    print(x, x.shape)
    print(torch.sum(torch.abs(x) - torch.abs(y)), torch.sum(x - y))


def neg_RESCAL():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    neg_size = Var(name='neg_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    NEemb = Tensor((nnodes, dim, neg_size), name='NEemb')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]

    mt = NEemb[t]

    res = bvv(bvm(vh, mr), vt)
    # res = bvm(bvm(vh, mr), mt)
    # code = codegen.cpu.print_cpp(gen_ir(res))
    # print(code)
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    batchsize=1024
    dimension=512
    relations = 30
    entities = 999
    negsize = 64
    hh = torch.randint(0, entities, (batchsize, )).unsqueeze(1).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).unsqueeze(1).cuda(0)
    tt = torch.randint(0, entities, (batchsize, negsize)).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension, dimension)).cuda(0)

    tt = tt.reshape(-1)
    hh = hh.repeat(1,negsize).reshape(-1)
    rr = rr.repeat(1,negsize).reshape(-1)

    # y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[hh], remb[rr]), eemb[tt])
    # print(y, y.shape)

    x = run.gpu.compile_and_run(code, batchsize*negsize, dimension, 0, eemb, hh, 0, remb, rr, tt)
    print(x, x.shape)


def backward_transr():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    score_emb = Tensor((batch_size, dim), name='score_emb')
    b_norm = Tensor((batch_size, ), name='b_norm')
    grad = Tensor((batch_size, ), name='dout')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]
    vr = Remb[r]

    res = bvm(vh - vt, mr) + vr
    code = codegen.gpu.print_cuda(gen_ir(res))
    # write_fwd(code)

    norm2 = norm(score_emb, 2, -1)
    # write_norm(codegen.gpu.print_cuda(gen_ir(norm2)))

    bwd_h = bsv(grad/b_norm, bvm(score_emb, mr))
    # write_bwd_h(codegen.gpu.print_cuda(gen_ir(bwd_h)))

    bwd_pr = bsm(grad/b_norm, bov(score_emb, vh-vt))
    # write_bwd_pr(codegen.gpu.print_cuda(gen_ir(bwd_pr)))

    batchsize=32
    dimension=64
    relations = 32
    entities = 99
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension, dimension)).cuda(0)

    eemb.requires_grad = True
    remb.requires_grad = True
    pemb.requires_grad = True

    
    func = transr_func.apply
    y = func(batchsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, remb)

    loss = get_loss(y, y)
    loss.backward()
    # print(eemb.grad, remb.grad, pemb.grad)
    y1 = eemb.grad
    y2 = remb.grad
    y3 = pemb.grad


    y = torch.norm(torch.einsum('ij,ijk->ik', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr], p=2, dim=-1)
    loss = get_loss(y,y)
    loss.backward()
    x1 = eemb.grad
    x2 = remb.grad
    x3 = pemb.grad

    print(torch.sum(torch.abs(x1) - torch.abs(y1)), torch.sum(torch.abs(x2) - torch.abs(y2)), torch.sum(torch.abs(x3) - torch.abs(y3)))

def backward_rescal():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    grad = Tensor((batch_size, ), name='dout', )
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]
    vr = Remb[r]

    res = bvv(bvm(vh, mr), vt)
    # code = codegen.gpu.print_cuda(gen_ir(res))
    # write_fwd(codegen.gpu.print_cuda(gen_ir(res)))

    bwd_h = bsv(grad, bvm(vt, mr))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_h)))
    # write_bwd_h(codegen.gpu.print_cuda(gen_ir(bwd_h)))

    bwd_t = bsv(grad, bvm(vh, mr))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_t)))
    # write_bwd_t(codegen.gpu.print_cuda(gen_ir(bwd_t)))

    bwd_r = bsm(grad, bov(vh, vt))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_r)))
    # write_bwd_r(codegen.gpu.print_cuda(gen_ir(bwd_r)))

    batchsize=32
    dimension=64
    relations = 32
    entities = 99
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension, dimension)).cuda(0)

    eemb.requires_grad = True
    remb.requires_grad = True
    pemb.requires_grad = True

    
    func = rescal_func.apply
    y = func(batchsize, dimension, 0, eemb, hh, 0, pemb, rr, tt)

    loss = get_loss(y, y)
    loss.backward()
    print(eemb.grad.shape, pemb.grad.shape)
    x1 = eemb.grad
    x2 = pemb.grad

    y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[hh], pemb[rr]), eemb[tt])
    loss = get_loss(y, y)
    loss.backward()
    x3 = eemb.grad
    x4 = pemb.grad
    print(torch.sum(torch.abs(x1) - torch.abs(x3)), torch.sum(torch.abs(x2) - torch.abs(x4)))

def backward_transh():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim), name='Proj')
    score_emb = Tensor((batch_size, dim), name='score_emb')
    b_norm = Tensor((batch_size, ), name='b_norm')
    grad = Tensor((batch_size, ), name='dout')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vp = Proj[r]
    vr = Remb[r]
    one = Const(1, 'int')

    res = vh - vt + vr - bsv(bvv(vp, vh - vt), vp)
    # code = codegen.gpu.print_cuda(gen_ir(res))
    write_fwd(codegen.gpu.print_cuda(gen_ir(res)))

    norm2 = norm(score_emb, 2, -1)
    write_norm(codegen.gpu.print_cuda(gen_ir(norm2)))

    # bwd_h = bvm(grad, one+bov(vp, vp))
    bwd_h = bsv(grad/b_norm, score_emb + bsv(bvv(vp, score_emb), vp))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_h)))
    write_bwd_h(codegen.gpu.print_cuda(gen_ir(bwd_h)))

    bwd_r = bsv(grad / b_norm, score_emb)
    write_bwd_r(codegen.gpu.print_cuda(gen_ir(bwd_r)))

    # bwd_pr = bvm(grad, bov(vp, vh-vt) + bvv(vh-vt, vp))
    bwd_pr = bsv(grad / b_norm, bsv(bvv(score_emb, vp), vh-vt) + bsv(bvv(vh-vt, vp), score_emb))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_pr)))
    write_bwd_pr(codegen.gpu.print_cuda(gen_ir(bwd_pr)))

    batchsize=128
    dimension=64
    relations = 32
    entities = 99
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension)).cuda(0)

    eemb.requires_grad = True
    remb.requires_grad = True
    pemb.requires_grad = True

    
    func = transh_func.apply
    y = func(batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, pemb)

    loss = get_loss(y, y)
    loss.backward()
    print(eemb.grad.shape, pemb.grad.shape, remb.grad.shape)
    x1 = eemb.grad
    x2 = pemb.grad
    x3 = remb.grad

    y = eemb[hh] - eemb[tt] + remb[rr] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', pemb[rr], eemb[hh]-eemb[tt]), pemb[rr])
    loss = get_loss(y, y)
    loss.backward()
    x4 = eemb.grad
    x5 = pemb.grad
    x6 = remb.grad
    print(torch.sum(torch.abs(x1) - torch.abs(x4)), torch.sum(torch.abs(x2) - torch.abs(x5)), torch.sum(torch.abs(x3) - torch.abs(x6)))

def backward_transf():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim), name='Proj')
    grad = Tensor((batch_size, ), name='dout', )
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vp = Proj[r]
    vr = Remb[r]
    two = Const(2, 'int')

    res = res = bvv(vh+vr, vt) + bvv(vt-vr, vh)
    code = codegen.gpu.print_cuda(gen_ir(res))
    # write_fwd(code)

    bwd_h = bsv(grad, two*vt-vr)
    # print(codegen.gpu.print_cuda(gen_ir(bwd_h)))
    # write_bwd_h(codegen.gpu.print_cuda(gen_ir(bwd_h)))

    bwd_t = bsv(grad, two*vh+vr)
    # print(codegen.gpu.print_cuda(gen_ir(bwd_t)))
    # write_bwd_t(codegen.gpu.print_cuda(gen_ir(bwd_t)))

    bwd_r = bsv(grad, vt-vh)
    # print(codegen.gpu.print_cuda(gen_ir(bwd_r)))
    # write_bwd_r(codegen.gpu.print_cuda(gen_ir(bwd_r)))

    batchsize=32
    dimension=64
    relations = 32
    entities = 99
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)
    pemb = torch.rand((relations, dimension, dimension)).cuda(0)

    eemb.requires_grad = True
    remb.requires_grad = True
    pemb.requires_grad = True

    
    func = transf_func.apply
    y = func(batchsize, dimension, 0, eemb, hh, 0, remb, rr, tt)

    loss = get_loss(y, y)
    loss.backward()
    print(eemb.grad.shape, remb.grad.shape)

def backward_transe():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb', )
    grad = Tensor((batch_size, dim), name='dout', )
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = vh - vt + vr
    code = codegen.gpu.print_cuda(gen_ir(res))

    batchsize=1024
    dimension=512
    relations = 30
    entities = 999
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)

    eemb.requires_grad = True
    remb.requires_grad = True

    # write_fwd(code)
    func = transe_func.apply
    y = func(batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr)

    loss = get_loss(y, y)
    loss.backward()
    print(y.grad, eemb.grad, remb.grad)


if __name__ == "__main__":
    # transE()
    # transH()
    # transR()
    # transF()
    # RESCAL()
    # backward()
    # neg_TransE()
    # neg_TransH()
    # neg_TransF()
    # neg_TransR()
    # neg_RESCAL()
    backward_transr()
    # backward_rescal()
    # backward_transh()
    # backward_transf()
    # backward_transe()