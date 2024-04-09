import transform
import run
from codegen import *
from helpers import ASGTraversal, IRTraversal, flatten, get_obj
from transform.fuse import basic_rule, fuse_operators
from asg import *
from asg2ir import gen_ir
from ir import *
import os

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

@new_op
def bsm(a, b):
    return apply(lambda x, y: einsum(',ij->ij', x, y), (a, b))

@new_op
def bmm(a,b):
    return apply(lambda x, y: einsum('ik,kj->j', x, y), (a,b))


def fuse_rule(node, res):
    '''
    Our specific loop fusion strategies for KGE.
    We can fuse some specific operators to reduce memory access and other overhead.

    Parameters:
    node (asg.ASTNode): root node of user-defined operation.
    '''
    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bvv':
        # fuse bvv and sub-operatos (elementwise or bvm)
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op or ('op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] == 'bvm'):
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op or ('op_name' in node.operators[2].attr and node.operators[2].attr['op_name'] == 'bvm'):
                fuse_operators(node, node.input_orders[2], node.operators[2])

    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] in ['bsv']:
        # fuse bsv and sub-operators (elementwise or bvv or sum) and (elementwise or bvm)
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op or ('op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] in ['bvv', 'sum']):
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op or ('op_name' in node.operators[2].attr and node.operators[2].attr['op_name'] == 'bvm'):
                fuse_operators(node, node.input_orders[2], node.operators[2])
        
    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bsm':
        # fuse bsm and sub-operators (elementwise or bvv) and (elementwise or bov)
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op or ('op_name' in node.operators[1].attr and node.operators[1].attr['op_name'] == 'bvv'):
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op or ('op_name' in node.operators[2].attr and node.operators[2].attr['op_name'] == 'bov'):
                fuse_operators(node, node.input_orders[2], node.operators[2])

    if type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'bov':
        # fuse bov and sub-operators (elementwise)
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[2], node.operators[2])

class fuser:
    '''
    Transformation pass for loop fusion for the given asgnode.
    basic_rule is default fusion rule in cuKE, fuse_rule is our specific loop fusion strategies.
    '''
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
    '''
    Transformation pass for loop tiling and parallelization for the given asgnode.
    We tile 3-level loops and apply different parallel hierarchy to each loop for parallelization.
    '''
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

        t = ASGTraversal(action)
        t(node)
        return node

class smem():
    '''
    Transformation pass for adding memory optimization for the given asgnode.
    We will move intermediate results into GPU shared memory to reduce global memory traffic.
    '''
    def __init__(self, C, D):
        self.C = C
        self.D = D

    def __call__(self, node):
        def action(n, res):
            if type(n) == TensorOp and 'op_name' in n.attr and n.attr['op_name'] in ['bsv', 'bvv', 'bvm', 'sum']:
                
                if n.compute:
                    transform.cuda_smem.add_direct_cache(node, n.eval)
                else:
                    def action(nn, res):
                        transform.cuda_smem.add_direct_cache(nn, n.eval)
                        if nn.ref_by:
                            for i in nn.ref_by:
                                    transform.cuda_smem.add_direct_cache(i, n.eval)
                    ASGTraversal(action)(node)
                # this function should 1) create a shared memory tensor for n.eval, 2) change all reference to the shared memory tensor for code in the scope of node, 3) handle both reduction ore non-reduction data
            
            if type(n) == TensorOp and n.op_type == 'index' and 'reuse' in n.operators[1].attr and n.operators[1].attr['reuse'] == True:
                unique_idx = Tensor((n.operators[1]._size()[0]/self.C, self.C), dtype='int', name=n.operators[1].name+'_uniq')
                buf_idx = Tensor((n.operators[1]._size()[0]/self.C, self.C), dtype='int', name=n.operators[1].name+'_buf')
                unique_cnt = Tensor((n.operators[1]._size()[0]/self.C, ), dtype='int', name=n.operators[1].name+'_unique_cnt')
                n.operators[1].attr['idx'] = [[unique_idx.name, unique_idx], [buf_idx.name, buf_idx], [unique_cnt.name, unique_cnt]]
                
                # this function should 1) create a shared memory tensor for n.eval, 2) analyze n.eval.idx to get buf_idx/uniq_idx, 3) change all reference based on buf_idx/uniq_idx for code in the scope of node

                if n.compute:
                    transform.cuda_smem.add_indirect_cache(node, n, self.C, self.D, unique_idx, buf_idx, unique_cnt)
                else:
                    def action(nn, res):
                        transform.cuda_smem.add_indirect_cache(nn, n, self.C, self.D, unique_idx, buf_idx, unique_cnt)
                        if nn.ref_by:
                            for i in nn.ref_by:
                                    transform.cuda_smem.add_indirect_cache(i, n, self.C, self.D, unique_idx, buf_idx, unique_cnt)
                    ASGTraversal(action)(node)

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

            if type(n) == TensorOp and 'op_name' in n.attr and n.attr['op_name'] in ['bvv', 'sum']:
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


transform.passes = [fuser(), tiler(16, 64), smem(16, 64)]


def write_code(code, filename):
    '''
    Write generated code into file.
    If file contents are the same with the generated code, no texts will be written into the file.
    '''
    filename = os.path.join('run/.tmp/', filename)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
            if content != code:
                f = open(filename, 'w')
                f.write(code)
                f.close()
    else:
        f = open(filename, 'w')
        f.write(code)
        f.close()

def inspector(r, rel_num, C = 16):
    '''
    This is our runtime inspection for index building.

    Parameters:
    r (tensor): relation indices.
    rel_num (int): number of edges.
    C (int): group size of each chunk.

    Returns:
    uniq (tensor): unique indices for reuse relations.
    buf (tensor): buffer indices for relation indices.
    cnt (tensor): count number of reuse relations for each group with size C.
    '''

    uniq = torch.zeros((r.shape[0]//C, C), dtype=torch.int64).cuda(0)
    buf = torch.zeros((r.shape[0]//C, C), dtype=torch.int64).cuda(0)
    cnt = torch.zeros((r.shape[0]//C, ), dtype=torch.int64).cuda(0)

    module = load(name='sort', sources=['apps/kge/inspection/inspector.cu'])
    module.gpu_sort(r, uniq, buf, cnt, int(r.shape[0]), int(rel_num))

    return uniq, buf, cnt

def transE():
    # We need to define all the arguments we need for our computation
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
    
    # TransE: Eemb[h] - Eemb[t] + Remb[r]
    res = vh - vt + vr
    # code = codegen.cpu.print_cpp(gen_ir(res))
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    # Here our cuda code is then generated, next step is create tensors to run the kernel
    batchsize=256
    dimension=512
    relations = 1800
    entities = 999
    neg_size = 64
    hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    eemb = torch.rand((entities, dimension)).cuda(0)
    remb = torch.rand((relations, dimension)).cuda(0)

    # for no r reuse, we need to comment r.attr['reuse'] = True
    # y = torch.norm(eemb[hh] - eemb[tt] + remb[rr], p=2, dim=-1)
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))

    # reuse sort and index building
    indices = torch.argsort(rr)
    hh = hh[indices]
    tt = tt[indices]
    rr = rr[indices]
    uniq, buf, cnt = inspector(rr, relations)
    y = eemb[hh] - eemb[tt] + remb[rr]
    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, uniq, buf, cnt)
    print(torch.sum(torch.abs(x) - torch.abs(y)))



def transH():
    # We need to define all the arguments we need for our computation
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

    # TransH: Eemb[h] - Eemb[t] + Remb[r] - Pemb[r]^T * (Eemb[h] - Eemb[t]) * Pemb[r]
    res = vh - vt + vr - bsv(bvv(vp, vh - vt), vp)
    # code = codegen.cpu.print_cpp(gen_ir(res))
    code = codegen.gpu.print_cuda(gen_ir(res))
    # print(code)

    # cuda code is then generated, next step is create tensors to run the kernel

    # batchsize=512
    # dimension=512
    # relations = 51
    # entities = 9999
    # hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    # rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    # tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    # eemb = torch.rand((entities, dimension)).cuda(0)
    # remb = torch.rand((relations, dimension)).cuda(0)
    # pemb = torch.rand((relations, dimension)).cuda(0)

    # for no r reuse
    # y = eemb[hh] - eemb[tt] + remb[rr] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', pemb[rr], eemb[hh]-eemb[tt]), pemb[rr])
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, pemb)

    # reuse sort and index building
    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]
    # uniq, buf, cnt = inspector(rr, relations)
    # y = y = eemb[hh] - eemb[tt] + remb[rr] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', pemb[rr], eemb[hh]-eemb[tt]), pemb[rr])
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr, uniq, buf, cnt, pemb)
    # print(torch.sum(torch.abs(x) - torch.abs(y)), torch.sum(x - y))



def transR():
    # We need to define all the arguments we need for our computation
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

    # TransR: (Eemb[h] - Eemb[t])^T * Proj[r] + Remb[r]
    res = bvm(vh - vt, mr) + vr
    # code = codegen.cpu.print_cpp(gen_ir(res))
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    # batchsize=512
    # dimension=256
    # relations = 30
    # entities = 9999
    # hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    # rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    # tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    # eemb = torch.rand((entities, dimension)).cuda(0)
    # remb = torch.rand((relations, dimension)).cuda(0)
    # pemb = torch.rand((relations, dimension, dimension)).cuda(0)

    # for no r reuse
    # y = torch.einsum('ab,abc->ac', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr]
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, remb)

    # reuse sort and index building
    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]
    # uniq, buf, cnt = inspector(rr, relations)
    # y = torch.einsum('ab,abc->ac', eemb[hh] - eemb[tt], pemb[rr]) + remb[rr]
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, pemb, rr, uniq, buf, cnt, remb)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))


def transF():
    # We need to define all the arguments we need for our computation
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

    # TransF: (Eemb[h] + Remb[r])^T * Eemb[t] + (Eemb[t] - Remb[r])^T * Eemb[h]
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

    # # for no r reuse
    # # y = torch.einsum('ab,ab->a', eemb[hh], eemb[tt]) - torch.einsum('ab,ab->a',(eemb[hh] - eemb[tt]), remb[rr])
    # # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, tt, 0, remb, rr)

    # # reuse sort and index building
    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]
    # uniq, buf, cnt = inspector(rr, relations)
    # # y = torch.einsum('ab,ab->a', eemb[hh], eemb[tt]) - torch.einsum('ab,ab->a',(eemb[hh] - eemb[tt]), remb[rr])
    # y = torch.einsum('ab,ab->a', eemb[hh] + remb[rr], eemb[tt]) + torch.einsum('ab,ab->a',(eemb[tt] - remb[rr]), eemb[hh])
    # x = run.gpu.compile_and_run(code, batchsize, dimension, entities, eemb, hh, relations, remb, rr, uniq, buf, cnt, tt)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))


def RESCAL():
    # We need to define all the arguments we need for our computation
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

    # RESCAL: Eemb[h]^T * Remb[r] * Eemb[t]
    res = bvv(bvm(vh, mr), vt)
    # code = codegen.cpu.print_cpp(gen_ir(res))
    code = codegen.gpu.print_cuda(gen_ir(res))
    print(code)

    # batchsize=512
    # dimension=512
    # relations = 30
    # entities = 999
    # hh = torch.randint(0, entities, (batchsize, )).cuda(0)
    # rr = torch.randint(0, relations, (batchsize, )).cuda(0)
    # tt = torch.randint(0, entities, (batchsize, )).cuda(0)
    # eemb = torch.rand((entities, dimension)).cuda(0)
    # remb = torch.rand((relations, dimension, dimension)).cuda(0)

    # # for no r reuse
    # # y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[hh], remb[rr]), eemb[tt])
    # # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, 0, remb, rr, tt)

    # # reuse sort and index building
    # indices = torch.argsort(rr)
    # hh = hh[indices]
    # tt = tt[indices]
    # rr = rr[indices]
    # uniq, buf, cnt = inspector(rr, relations)
    # y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[hh], remb[rr]), eemb[tt])
    # x = run.gpu.compile_and_run(code, batchsize, dimension, 0, eemb, hh, 0, remb, rr, uniq, buf, cnt, tt)
    # print(torch.sum(torch.abs(x) - torch.abs(y)))


def backward_transr():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    grad = Tensor((batch_size, dim), name='dout')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]
    vr = Remb[r]

    # gradient of vh is dout^T * mr
    # gradient of vt is - dout^T * mr
    bwd_h = bvm(grad, mr)
    # print(codegen.gpu.print_cuda(gen_ir(bwd_h)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_h)), 'bwd_h.cu')

    # gradient of mr is dout * (vh - vt)^T
    bwd_pr = bov(grad, vh-vt)
    # print(codegen.gpu.print_cuda(gen_ir(bwd_pr)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_pr)), 'bwd_pr.cu')


def backward_rescal():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Proj = Tensor((nedges, dim, dim), name='Proj')
    grad = Tensor((batch_size, ), name='dout', )
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]

    # gradient of vh is dout * (vt^T * mr)
    bwd_h = bsv(grad, bvm(vt, mr))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_h)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_h)), 'bwd_h.cu')

    # gradient of vt is dout * (vh^T * mr)
    bwd_t = bsv(grad, bvm(vh, mr))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_t)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_t)), 'bwd_t.cu')

    # gradient of mr is dout * (vh * vt^T)
    bwd_r = bsm(grad, bov(vh, vt))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_r)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_r)), 'bwd_r.cu')


def backward_transh():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    Proj = Tensor((nedges, dim), name='Proj')
    grad = Tensor((batch_size, dim), name='dout')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vp = Proj[r]
    vr = Remb[r]
    one = Const(1, 'int')

    # gradient of vh is dout^T * (1 + vp * vp^T)
    # gradient of vt is - dout^T * (1 + vp * vp^T)
    bwd_h = bvm(grad, one+bov(vp, vp))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_h)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_h)), 'bwd_h.cu')

    # gradient of vp is dout^T * (vp * (vh - vt)^T + (vh - vt)^T * vp)
    bwd_pr = bvm(grad, bov(vp, vh-vt) + bvv(vh-vt, vp))
    # print(codegen.gpu.print_cuda(gen_ir(bwd_pr)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_pr)), 'bwd_pr.cu')


def backward_transf():
    nnodes = Var(name='nnodes')
    nedges = Var(name='nedges')
    dim = Var(name='dim')
    batch_size = Var(name='batch_size')
    Eemb = Tensor((nnodes, dim), name='Eemb')
    Remb = Tensor((nedges, dim), name='Remb')
    grad = Tensor((batch_size, ), name='dout')
    h = Tensor((batch_size, ), dtype='int', name='h')
    t = Tensor((batch_size, ), dtype='int', name='t')
    r = Tensor((batch_size, ), dtype='int', name='r')
    # r.attr['reuse'] = True
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    two = Const(2, 'int')
    
    # gradient of vh is dout * (2 * vt - vr)
    bwd_h = bsv(grad, two*vt-vr)
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_h)), 'bwd_h.cu')

    # gradient of vt is dout * (2 * vh + vr)
    bwd_t = bsv(grad, two*vh+vr)
    # print(codegen.gpu.print_cuda(gen_ir(bwd_t)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_t)), 'bwd_t.cu')

    # gradient of vr is dout * (vt - vh)
    bwd_r = bsv(grad, vt-vh)
    # print(codegen.gpu.print_cuda(gen_ir(bwd_r)))
    # write_code(codegen.gpu.print_cuda(gen_ir(bwd_r)), 'bwd_r.cu')



if __name__ == "__main__":
    transE()
    # transH()
    # transR()
    # transF()
    # RESCAL()
    # backward_transr()
    # backward_rescal()
    # backward_transh()
    # backward_transf()
    # backward_transe()