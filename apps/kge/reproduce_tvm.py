import tvm
from tvm import relay, te
from tvm.contrib import graph_executor
import torch
import argparse
from apps.kge.data import *

parser = argparse.ArgumentParser(description="test on pytorch")

parser.add_argument('--model', type=str, default='TransE', help='The models.')
parser.add_argument('--batch_size', type=int, default=1024, help='The batch size used for validation and test.')
parser.add_argument('--dim', type=int, default=512, help='The embedding size of relation and entity.')
parser.add_argument('--dataset', type=str, default='FB15k', help='The name of the builtin knowledge graph. cuKE automatically downloads the knowledge graph and keep it under data_path.')
parser.add_argument('--neg_sample_size', type=int, default=64, help='The number of negative samples we use for each positive sample in the training.')

args = parser.parse_args()

normal_shape = (args.batchsize, 1, args.dim)
neg_shape = (args.batchsize, args.neg_sample_size, args.dim)

target = tvm.target.Target('cuda')
dev = tvm.cuda()

def transR(PROJ, EMB, RELEMB, VH, VT, VR):
    RES = relay.nn.batch_matmul((relay.adv_index([EMB, VH]) - relay.adv_index([EMB, VT])), relay.adv_index([PROJ, VR]), transpose_a=False, transpose_b=False) + relay.adv_index([RELEMB, VR])

    return RES

def transR_neg(PROJ, EMB, RELEMB, VH, VT, VR):
    H = relay.adv_index([EMB, VH])
    H = relay.reshape(H, newshape=normal_shape)
    H = relay.op.broadcast_to(H, shape=neg_shape)
    T = relay.reshape(relay.adv_index([EMB, VT]), newshape=neg_shape)
    HT = relay.nn.batch_matmul(H - T, relay.adv_index([PROJ, VR]), transpose_a=False, transpose_b=False)
    R = relay.adv_index([RELEMB, VR])
    R = relay.reshape(R, newshape=normal_shape)
    R = relay.op.broadcast_to(R, shape=neg_shape)
    
    RES=HT+R

    return RES

def transH(PROJ, EMB, RELEMB, VH, VT, VR):
    H = relay.adv_index([EMB, VH])
    R = relay.adv_index([RELEMB, VR])
    T = relay.adv_index([EMB, VT])
    Pv = relay.adv_index([PROJ, VR])

    ht = H-T
    wh = relay.squeeze(relay.nn.batch_matmul(relay.expand_dims(ht, axis=1), relay.expand_dims(Pv, axis=1)), axis=1)
    
    h = relay.multiply(wh, Pv)
    RES = H - h + R - T
    return RES

def transH_neg(PROJ, EMB, RELEMB, VH, VT, VR):
    H = relay.adv_index([EMB, VH])
    H1 = relay.op.broadcast_to(relay.reshape(H, newshape=normal_shape), shape=neg_shape)
    R = relay.adv_index([RELEMB, VR])
    R1 = relay.op.broadcast_to(relay.reshape(R, newshape=normal_shape), shape=neg_shape)
    T = relay.reshape(relay.adv_index([EMB, VT]), newshape=neg_shape)
    Pv = relay.adv_index([PROJ, VR])
    Pv1 = relay.expand_dims(Pv, axis=2)
    w = relay.nn.batch_matmul(H1-T, Pv1, transpose_a=False, transpose_b=False)
    w = relay.nn.batch_matmul(w, relay.expand_dims(Pv, axis=1), transpose_a=False, transpose_b=False)
    
    RES = H1 + R1 - T - w
    return RES

def RESCAL(EMB, RELEMB, VH, VT, VR):
    RES = relay.nn.batch_matmul(relay.nn.batch_matmul(relay.adv_index([EMB, VH]), relay.adv_index([RELEMB, VR]), transpose_a=False, transpose_b=False), relay.adv_index([EMB, VT]))
    return RES

def RESCAL_neg(EMB, RELEMB, VH, VT, VR):
    H = relay.reshape(relay.adv_index([EMB, VH]), newshape=normal_shape)
    T = relay.reshape(relay.adv_index([EMB, VT]), newshape=neg_shape)
    RES = relay.nn.batch_matmul(relay.nn.batch_matmul(H, relay.adv_index([RELEMB, VR]), transpose_a=False, transpose_b=False), T)
    return RES

def transF(EMB, RELEMB, VH, VT, VR):
    H = relay.adv_index([EMB, VH])
    R = relay.adv_index([RELEMB, VR])
    T = relay.adv_index([EMB, VT])

    RES = relay.multiply(relay.prod(H*T, axis=1), relay.const(2, dtype="float32")) + relay.prod((H-T)*R, axis=1)
    return RES

def TransF_neg(EMB, RELEMB, VH, VT, VR):
    H = relay.adv_index([EMB, VH])
    R = relay.adv_index([RELEMB, VR])

    T = relay.adv_index([EMB, VT])
    H = relay.expand_dims(H, axis=1)
    R = relay.expand_dims(R, axis=1)
    RES = relay.squeeze(relay.multiply(relay.nn.batch_matmul(H, T), relay.const(2, dtype="float32")), axis=1) + relay.squeeze(relay.nn.batch_matmul(H - T, R), axis=2)

    return RES


def transE(EMB, RELEMB, VH, VT, VR):
    H = relay.adv_index([EMB, VH])
    R = relay.adv_index([RELEMB, VR])
    T = relay.adv_index([EMB, VT])

    RES = H+R-T
    return RES

def transE_neg(EMB, RELEMB, VH, VT, VR):
    HR = relay.adv_index([EMB, VH]) + relay.adv_index([RELEMB, VR])
    HR = relay.reshape(HR, newshape=normal_shape)
    HR = relay.op.broadcast_to(HR, shape=neg_shape)
    RES = HR - relay.adv_index([EMB, VT])
    return RES



def TransE_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]

    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transE(EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())

def TransR_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]

    PROJ = relay.var('proj', shape=(nrel, dim, dim), dtype='float32')
    EMB = relay.var('emb', shape=(nnodes, 1, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, 1, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transR(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    tproj = tvm.nd.array(projection_emb)
    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('proj', tproj)
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)
    
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())


def RESCAL_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]
    EMB = relay.var('emb', shape=(nnodes, 1, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = RESCAL(EMB, RELEMB, VH, VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)

    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())

def TransH_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]
    PROJ = relay.var('proj', shape=(nrel, dim), dtype='float32')
    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transH(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)
    tproj = tvm.nd.array(projection_emb)
    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('proj', tproj)
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)
    
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())

def TransF_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]

    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = transF(EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)
    
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())
    
def neg_TransE_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]

    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize, 64), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transE_neg(EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(neg_tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)

    print(m.benchmark(dev, number=1, repeat=1))
    neg_score = torch.from_numpy(m.get_output(0).asnumpy())

def neg_TransR_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]

    PROJ = relay.var('proj', shape=(nrel, dim, dim), dtype='float32')
    EMB = relay.var('emb', shape=(nnodes, 1, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, 1, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize * 64, ), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = transR_neg(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)
    tproj = tvm.nd.array(projection_emb)
    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(neg_tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('proj', tproj)
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)

    print(m.benchmark(dev, number=1, repeat=1))
    neg_score = torch.from_numpy(m.get_output(0).asnumpy())

def neg_RESCAL_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]
    EMB = relay.var('emb', shape=(nnodes, 1, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize*64,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = RESCAL_neg(EMB, RELEMB, VH, VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(neg_tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)

    print(m.benchmark(dev, number=1, repeat=1))
    neg_score = torch.from_numpy(m.get_output(0).asnumpy())

def neg_TransH_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]
    PROJ = relay.var('proj', shape=(nrel, dim), dtype='float32')
    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize * 64, ), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = transH_neg(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)
    tproj = tvm.nd.array(projection_emb)
    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(neg_tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('proj', tproj)
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)

    print(m.benchmark(dev, number=1, repeat=1))
    neg_score = torch.from_numpy(m.get_output(0).asnumpy())

def neg_TransF_tvm(batchsize, dim):
    samplers, entity_emb, relation_emb, projection_emb = get_samplers(args)
    rel_ids, head_ids, tail_ids, neg_rel_ids, neg_head_ids, neg_tail_ids = get_indices(args, samplers)

    nrel = relation_emb.shape[0]
    nnodes = entity_emb.shape[0]

    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,64,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = TransF_neg(EMB, RELEMB, VH, VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(entity_emb)
    trelemb = tvm.nd.array(relation_emb)
    tvh = tvm.nd.array(head_ids)
    tvt = tvm.nd.array(neg_tail_ids)
    tvr = tvm.nd.array(rel_ids)

    params = {}

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)

    print(m.benchmark(dev, number=1, repeat=1))
    neg_score = torch.from_numpy(m.get_output(0).asnumpy())

if __name__ == '__main__':
    if args.model == 'TransE':
        TransE_tvm(args.batchsize, args.dim)
    elif args.model == 'TransR':
        TransR_tvm(args.batchsize, args.dim)
    elif args.model == 'TransH':
        TransH_tvm(args.batchsize, args.dim)
    elif args.model == 'TransF':
        TransF_tvm(args.batchsize, args.dim)
    elif args.model == 'RESCAL':
        RESCAL_tvm(args.batchsize, args.dim)
    elif args.model == 'neg_TransE':
        neg_TransE_tvm(args.batchsize, args.dim)
    elif args.model == 'neg_TransR':
        neg_TransR_tvm(args.batchsize, args.dim)
    elif args.model == 'neg_TransH':
        neg_TransH_tvm(args.batchsize, args.dim)
    elif args.model == 'neg_TransF':
        neg_TransF_tvm(args.batchsize, args.dim)
    elif args.model == 'neg_RESCAL':
        neg_RESCAL_tvm(args.batchsize, args.dim)
    