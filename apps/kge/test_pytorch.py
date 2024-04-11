import torch
from data import *

import argparse
parser = argparse.ArgumentParser(description="test on pytorch")

parser.add_argument('--model', type=str, default='TransE', help='The models.')
parser.add_argument('--batch_size', type=int, default=1024, help='The batch size used for validation and test.')
parser.add_argument('--dim', type=int, default=512, help='The embedding size of relation and entity.')
parser.add_argument('--dataset', type=str, default='FB15k', help='The name of the builtin knowledge graph. cuKE automatically downloads the knowledge graph and keep it under data_path.')
parser.add_argument('--neg_sample_size', type=int, default=64, help='The number of negative samples we use for each positive sample in the training.')

args = parser.parse_args()

def transE():
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    y = entity_emb[head_ids] - entity_emb[tail_ids] + relation_emb[rel_ids]
    
    start_event.record()
    for i in range(100):
        y = entity_emb[head_ids] - entity_emb[tail_ids] + relation_emb[rel_ids]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Pytorch model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))



def transH():
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    y = entity_emb[head_ids] - entity_emb[tail_ids] + relation_emb[rel_ids] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', projection_emb[rel_ids], entity_emb[head_ids]-entity_emb[tail_ids]), projection_emb[rel_ids])
    start_event.record()
    for i in range(100):
         y = entity_emb[head_ids] - entity_emb[tail_ids] + relation_emb[rel_ids] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', projection_emb[rel_ids], entity_emb[head_ids]-entity_emb[tail_ids]), projection_emb[rel_ids])
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Pytorch model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))



def transR():
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    y = torch.einsum('ab,abc->ac', entity_emb[head_ids] - entity_emb[tail_ids], projection_emb[rel_ids]) + relation_emb[rel_ids]
    start_event.record()
    for i in range(100):
        y = torch.einsum('ab,abc->ac', entity_emb[head_ids] - entity_emb[tail_ids], projection_emb[rel_ids]) + relation_emb[rel_ids]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Pytorch model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))

def transF():
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    y = torch.einsum('ab,ab->a', entity_emb[head_ids] + relation_emb[rel_ids], entity_emb[tail_ids]) + torch.einsum('ab,ab->a',(entity_emb[tail_ids] - relation_emb[rel_ids]), entity_emb[head_ids])
    start_event.record()
    for i in range(100):
        y = torch.einsum('ab,ab->a', entity_emb[head_ids] + relation_emb[rel_ids], entity_emb[tail_ids]) + torch.einsum('ab,ab->a',(entity_emb[tail_ids] - relation_emb[rel_ids]), entity_emb[head_ids])
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Pytorch model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))


def RESCAL():
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', entity_emb[head_ids], relation_emb[rel_ids]), entity_emb[tail_ids])
    start_event.record()
    for i in range(100):
        y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', entity_emb[head_ids], relation_emb[rel_ids]), entity_emb[tail_ids])
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Pytorch model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))



def neg_transR():
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    y = torch.einsum('ab,abc->ac', entity_emb[neg_head_ids] - entity_emb[neg_tail_ids], projection_emb[neg_rel_ids]) + relation_emb[neg_rel_ids]
    start_event.record()
    for i in range(100):
        y = torch.einsum('ab,abc->ac', entity_emb[neg_head_ids] - entity_emb[neg_tail_ids], projection_emb[neg_rel_ids]) + relation_emb[neg_rel_ids]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Pytorch model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))


def neg_transF():
    samplers, entity_emb, relation_emb, projection_emb = get_samplers()
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(samplers)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The first time execution should cost too much time, our profiling should begin after warm up
    y = torch.einsum('ab,ab->a', entity_emb[neg_head_ids] + relation_emb[neg_rel_ids], entity_emb[neg_tail_ids]) + torch.einsum('ab,ab->a',(entity_emb[neg_tail_ids] - relation_emb[neg_rel_ids]), entity_emb[neg_head_ids])
    start_event.record()
    for i in range(100):
        y = torch.einsum('ab,ab->a', entity_emb[neg_head_ids] + relation_emb[neg_rel_ids], entity_emb[neg_tail_ids]) + torch.einsum('ab,ab->a',(entity_emb[neg_tail_ids] - relation_emb[neg_rel_ids]), entity_emb[neg_head_ids])
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Pytorch model {} on {} dataset completed! batchsize:{} dim:{}\naverage time cost: {} ms'.format(args.model, args.dataset, args.batch_size, args.dim, elapsed_time_ms/100))

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