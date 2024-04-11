from pycuke import transform, run
from pycuke.codegen import *
from pycuke.helpers import ASGTraversal, IRTraversal, flatten, get_obj
from pycuke.transform.fuse import basic_rule, fuse_operators
from pycuke.asg import *
from pycuke.asg2ir import gen_ir
from pycuke.ir import *
import os

import torch
from torch.utils.cpp_extension import load

from data import *
from kge import *

import argparse
parser = argparse.ArgumentParser(description="test on pytorch")

parser.add_argument('--model', type=str, default='TransE', help='The models.')
parser.add_argument('--batch_size', type=int, default=1024, help='The batch size used for validation and test.')
parser.add_argument('--dim', type=int, default=512, help='The embedding size of relation and entity.')
parser.add_argument('--dataset', type=str, default='FB15k', help='The name of the builtin knowledge graph. cuKE automatically downloads the knowledge graph and keep it under data_path.')
parser.add_argument('--neg_sample_size', type=int, default=64, help='The number of negative samples we use for each positive sample in the training.')

args = parser.parse_args()

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
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    # no relation reuse
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
    print('model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))



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
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])

    # Before run the code, please check the file in run/.tmp/cude_code.cu to 
    # make sure each argument corresponds to the arguments in the main function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, tail_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, projection_emb)
    start_event.record()
    for i in range(100):
        x = run.gpu.compile_and_run(code, args.batch_size, args.dim, entity_emb.shape[0], entity_emb, head_ids, tail_ids, relation_emb.shape[0], relation_emb, rel_ids, uniq, buf, cnt, projection_emb)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))



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
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])
    
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
    print('model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))

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
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])
    
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
    print('model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))


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
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(rel_ids)
    head_ids = head_ids[indices]
    tail_ids = tail_ids[indices]
    rel_ids = rel_ids[indices]
    uniq, buf, cnt = inspector(rel_ids, relation_emb.shape[0])

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
    print('model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))



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
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(neg_rel_ids)
    neg_head_ids = neg_head_ids[indices]
    neg_tail_ids = neg_tail_ids[indices]
    neg_rel_ids = neg_rel_ids[indices]
    uniq, buf, cnt = inspector(neg_rel_ids, relation_emb.shape[0])
    
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
    print('model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))


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
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    # reuse relation: sort and index building, we need to add 'r.attr['reuse'] = True' before gen_ir 
    indices = torch.argsort(neg_rel_ids)
    neg_head_ids = neg_head_ids[indices]
    neg_tail_ids = neg_tail_ids[indices]
    neg_rel_ids = neg_rel_ids[indices]
    uniq, buf, cnt = inspector(neg_rel_ids, relation_emb.shape[0])

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
    print('model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))

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