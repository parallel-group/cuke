from cuke import transform, run
from cuke.codegen import *
from cuke.helpers import ASGTraversal, IRTraversal, flatten, get_obj
from cuke.transform.fuse import basic_rule, fuse_operators
from cuke.asg import *
from cuke.asg2ir import gen_ir
from cuke.ir import *
import os

from apps.kge.data import *

import torch
from torch.utils.cpp_extension import load

import argparse
parser = argparse.ArgumentParser(description="test on pytorch")

parser.add_argument('--model', type=str, default='TransE', help='The models.')
parser.add_argument('--batch_size', type=int, default=1024, help='The batch size used for validation and test.')
parser.add_argument('--dim', type=int, default=512, help='The embedding size of relation and entity.')
parser.add_argument('--dataset', type=str, default='FB15k', help='The name of the builtin knowledge graph. cuKE automatically downloads the knowledge graph and keep it under data_path.')
parser.add_argument('--neg_sample_size', type=int, default=64, help='The number of negative samples we use for each positive sample in the training.')


args = parser.parse_args()

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
            
        # print(codegen.gpu.to_string(node.compute))
        self.eval = node.eval
        t = ASGTraversal(action)(node)
        return node

class indirect_smem():
    '''
    Transformation pass for adding memory optimization for the given asgnode.
    We will load indirect memory access data into GPU shared memory to reuse indices and reduce global memory traffic.
    '''
    def __init__(self, C, D):
        self.C = C
        self.D = D

    def __call__(self, node):
        def action(n, res):
            if type(n) == TensorOp and n.op_type == 'index' and 'reuse' in n.operators[1].attr and n.operators[1].attr['reuse'] == True:
                unique_idx = Tensor((n.operators[1]._size()[0]/self.C, self.C), dtype='int', name=n.operators[1].name+'_uniq')
                buf_idx = Tensor((n.operators[1]._size()[0]/self.C, self.C), dtype='int', name=n.operators[1].name+'_buf')
                unique_cnt = Tensor((n.operators[1]._size()[0]/self.C, ), dtype='int', name=n.operators[1].name+'_unique_cnt')
                n.operators[1].attr['idx'] = [[unique_idx.name, unique_idx], [buf_idx.name, buf_idx], [unique_cnt.name, unique_cnt]]
                
                # this function should 1) create a shared memory tensor for n.eval, 2) analyze n.eval.idx to get buf_idx/uniq_idx, 3) change all reference based on buf_idx/uniq_idx for code in the scope of node

                # if n.compute:
                transform.cuda_smem.add_indirect_cache(node, n, self.C, self.D, unique_idx, buf_idx, unique_cnt)
                # else:
                # print(node.op_type, codegen.gpu.to_string(node.compute))
                def action(nn, res):
                    transform.cuda_smem.add_indirect_cache(nn, n, self.C, self.D, unique_idx, buf_idx, unique_cnt)
                    if nn.ref_by:
                        for i in nn.ref_by:
                            transform.cuda_smem.add_indirect_cache(i, n, self.C, self.D, unique_idx, buf_idx, unique_cnt)
                ASGTraversal(action)(node)

            
        # print(codegen.gpu.to_string(node.compute))
        self.eval = node.eval
        t = ASGTraversal(action)(node)
        return node

transform.passes = [fuser(), tiler(16, 64), smem(16, 64), indirect_smem(16, 64)]
# transform.passes = [fuser(), tiler(16, 64)]
# transform.passes = [fuser()]

def init_embddings(dataset):
    projection_emb = None
    entity_emb = torch.rand(dataset.n_entities, args.dim, dtype=torch.float32, device='cuda:0')
    if args.model == 'RESCAL':
        relation_emb = torch.rand(dataset.n_relations, args.dim, args.dim, dtype=torch.float32, device='cuda:0')
    else:
        relation_emb = torch.rand(dataset.n_relations, args.dim, dtype=torch.float32, device='cuda:0')
    
    if args.model == 'TransH':
        projection_emb = torch.rand(dataset.n_relations, args.dim, dtype=torch.float32, device='cuda:0')
    elif args.model == 'TransR':
        projection_emb = torch.rand(dataset.n_relations, args.dim, args.dim, dtype=torch.float32, device='cuda:0')

    return entity_emb, relation_emb, projection_emb

def get_samplers():
    '''
    This function is used to sample batches from dataset.
    '''
    dataset = get_dataset(args.dataset)
    entity_emb, relation_emb, projection_emb = init_embddings(dataset)
    train_data = TrainDataset(dataset)

    train_sampler_head = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='head',
                                                       num_workers=8,
                                                       shuffle=True,
                                                       exclude_positive=False)
    train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                    args.neg_sample_size,
                                                    args.neg_sample_size,
                                                    mode='tail',
                                                    num_workers=8,
                                                    shuffle=True,
                                                    exclude_positive=False)
    train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                    args.neg_sample_size, args.neg_sample_size,
                                                    True, dataset.n_entities)

    return train_sampler, entity_emb, relation_emb, projection_emb

def get_indices(train_sampler):
    pos_g, neg_g = next(train_sampler)

    rel_ids = pos_g.edata['id'].cuda(0)
    head_ids, tail_ids = pos_g.all_edges(order='eid')
    head_ids = pos_g.ndata['id'][head_ids].cuda(0)
    tail_ids = pos_g.ndata['id'][tail_ids].cuda(0)

    neg_rel_ids = rel_ids.reshape(-1, 1).repeat(1, args.neg_sample_size).reshape(-1).cuda(0)
    if neg_g.neg_head:
        neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
        neg_tail_ids = tail_ids.reshape(-1, 1).repeat(1, args.neg_sample_size).reshape(-1).cuda(0)
    else:
        neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
        neg_head_ids = head_ids.reshape(-1, 1).repeat(1, args.neg_sample_size).reshape(-1).cuda(0)

    return rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids



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
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    
    # TransE: Eemb[h] - Eemb[t] + Remb[r]
    res = vh - vt + vr
    code = gpu.print_cuda(gen_ir(res))

    # Here our cuda code is then generated, next step is sample node indices from input graph and create embeddings to run the kernel
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    # no relation reuse
    y = entity_emb[head_ids] - entity_emb[tail_ids] + relation_emb[rel_ids]
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, tail_ids, relation_emb.shape[0], relation_emb, rel_ids)

    start_event.record()
    for i in range(100):
        x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, tail_ids, relation_emb.shape[0], relation_emb, rel_ids)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\ntotal error: {}, average time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, torch.sum(torch.abs(x) - torch.abs(y)), elapsed_time_ms/100))



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
    code = gpu.print_cuda(gen_ir(res))

    # Here our cuda code is then generated, next step is sample node indices from input graph and create embeddings to run the kernel
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])

    y = entity_emb[head_ids] - entity_emb[tail_ids] + relation_emb[rel_ids] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', projection_emb[rel_ids], entity_emb[head_ids]-entity_emb[tail_ids]), projection_emb[rel_ids])

    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, tail_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, projection_emb)
    start_event.record()
    # for i in range(100):
    #     x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, tail_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, projection_emb)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\ntotal error: {}, average time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, torch.sum(torch.abs(x) - torch.abs(y)), elapsed_time_ms/100))



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
    code = gpu.print_cuda(gen_ir(res))

    # Here our cuda code is then generated, next step is sample node indices from input graph and create embeddings to run the kernel
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])
    y = torch.einsum('ab,abc->ac', entity_emb[head_ids] - entity_emb[tail_ids], projection_emb[rel_ids]) + relation_emb[rel_ids]
    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, 0, entity_emb, head_ids, tail_ids, 0, projection_emb, rel_ids, uniq, buf, cnt, relation_emb)
    start_event.record()
    for i in range(100):
        x = run.gpu.compile_and_run(code, args.batch_size, args.dim, 0, entity_emb, head_ids, tail_ids, 0, projection_emb, rel_ids, uniq, buf, cnt, relation_emb)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\ntotal error: {}, average time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, torch.sum(torch.abs(x) - torch.abs(y)), elapsed_time_ms/100))

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
    code = gpu.print_cuda(gen_ir(res))

    # Here our cuda code is then generated, next step is sample node indices from input graph and create embeddings to run the kernel
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])
    
    y = torch.einsum('ab,ab->a', entity_emb[head_ids] + relation_emb[rel_ids], entity_emb[tail_ids]) + torch.einsum('ab,ab->a',(entity_emb[tail_ids] - relation_emb[rel_ids]), entity_emb[head_ids])

    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, tail_ids)
    start_event.record()
    for i in range(100):
        x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, tail_ids)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\ntotal error: {}, average time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, torch.sum(torch.abs(x) - torch.abs(y)), elapsed_time_ms/100))


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
    code = gpu.print_cuda(gen_ir(res))

    # Here our cuda code is then generated, next step is sample node indices from input graph and create embeddings to run the kernel
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])

    y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', entity_emb[head_ids], relation_emb[rel_ids]), entity_emb[tail_ids])

    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, tail_ids)
    start_event.record()
    for i in range(100):
        x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, tail_ids)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\ntotal error: {}, average time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, torch.sum(torch.abs(x) - torch.abs(y)), elapsed_time_ms/100))



def neg_transR():
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
    code = gpu.print_cuda(gen_ir(res))

    # Here our cuda code is then generated, next step is sample node indices from input graph and create embeddings to run the kernel
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(neg_rel_ids)
    neg_head_ids = neg_head_ids[indices]
    neg_tail_ids = neg_tail_ids[indices]
    neg_rel_ids = neg_rel_ids[indices]
    uniq, buf, cnt = inspector(neg_rel_ids, relation_emb.shape[0])
    y = torch.einsum('ab,abc->ac', entity_emb[neg_head_ids] - entity_emb[neg_tail_ids], projection_emb[neg_rel_ids]) + relation_emb[neg_rel_ids]
    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, 0, entity_emb, neg_head_ids, neg_tail_ids, 0, projection_emb, neg_rel_ids, uniq, buf, cnt, relation_emb)
    start_event.record()
    for i in range(100):
        x = run.gpu.compile_and_run(code, args.batch_size, args.dim, 0, entity_emb, neg_head_ids, neg_tail_ids, 0, projection_emb, neg_rel_ids, uniq, buf, cnt, relation_emb)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\ntotal error: {}, average time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, torch.sum(torch.abs(x) - torch.abs(y)), elapsed_time_ms/100))


def neg_transF():
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
    code = gpu.print_cuda(gen_ir(res))

    # Here our cuda code is then generated, next step is sample node indices from input graph and create embeddings to run the kernel
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(neg_rel_ids)
    neg_head_ids = neg_head_ids[indices]
    neg_tail_ids = neg_tail_ids[indices]
    neg_rel_ids = neg_rel_ids[indices]
    uniq, buf, cnt = inspector(neg_rel_ids, relation_emb.shape[0])
    
    y = torch.einsum('ab,ab->a', entity_emb[neg_head_ids] + relation_emb[neg_rel_ids], entity_emb[neg_tail_ids]) + torch.einsum('ab,ab->a',(entity_emb[neg_tail_ids] - relation_emb[neg_rel_ids]), entity_emb[neg_head_ids])

    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, neg_head_ids, relation_emb.shape[0], relation_emb, neg_rel_ids, uniq, buf, cnt, neg_tail_ids)
    start_event.record()
    for i in range(100):
        x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, neg_head_ids, relation_emb.shape[0], relation_emb, neg_rel_ids, uniq, buf, cnt, neg_tail_ids)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\ntotal error: {}, average time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, torch.sum(torch.abs(x) - torch.abs(y)), elapsed_time_ms/100))

if __name__ == "__main__":
    if args.model == 'TransE':
        transE()
    elif args.model == 'TransH':
        transH()
    elif args.model == 'TransR':
        transR()
    elif args.model == 'TransF':
        transF()
    elif args.model == 'RESCAL':
        RESCAL()
    elif args.model == 'neg_TransR':
        neg_transR()
    elif args.model == 'neg_TransF':
        neg_transF()