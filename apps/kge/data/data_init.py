from apps.kge.data import *
import torch

def init_embddings(args, dataset):
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

def get_samplers(args):
    '''
    This function is used to sample batches from dataset.
    '''
    dataset = get_dataset(args.dataset)
    entity_emb, relation_emb, projection_emb = init_embddings(args, dataset)
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

def get_indices(args, train_sampler):
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