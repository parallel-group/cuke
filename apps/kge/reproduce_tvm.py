import tvm
from tvm import relay, te
from tvm.contrib import graph_executor
import torch
import argparse

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
    eemb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransE_l1/eemb_{dataset}_{batchsize}_{dim}.pt')
    remb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransE_l1/remb_{dataset}_{batchsize}_{dim}.pt')

    heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransE_l1/heads_{dataset}_{batchsize}_{dim}.pt')
    tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransE_l1/tails_{dataset}_{batchsize}_{dim}.pt')
    relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransE_l1/relations_{dataset}_{batchsize}_{dim}.pt')

    neg_heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransE_l1/heads_{dataset}_{batchsize}_{dim}.pt')
    neg_tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransE_l1/tails_{dataset}_{batchsize}_{dim}.pt')
    neg_relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransE_l1/relations_{dataset}_{batchsize}_{dim}.pt')

    nrel = remb.shape[0]
    nnodes = eemb.shape[0]

    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transE(EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(heads)
    tvt = tvm.nd.array(tails)
    tvr = tvm.nd.array(relations)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)
    print(lib.lib.imported_modules[0].get_source())
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())

    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize, 64), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transE_neg(EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(neg_heads)
    tvt = tvm.nd.array(neg_tails)
    tvr = tvm.nd.array(neg_relations)

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

    print(pos_score.shape, neg_score.shape)

def TransR_tvm(batchsize, dim):
    eemb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransR/eemb_{dataset}_{batchsize}_{dim}.pt')
    remb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransR/remb_{dataset}_{batchsize}_{dim}.pt')
    pemb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransR/pemb_{dataset}_{batchsize}_{dim}.pt')
    
    heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransR/heads_{dataset}_{batchsize}_{dim}.pt')
    tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransR/tails_{dataset}_{batchsize}_{dim}.pt')
    relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransR/relations_{dataset}_{batchsize}_{dim}.pt')

    neg_heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransR/heads_{dataset}_{batchsize}_{dim}.pt')
    neg_tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransR/tails_{dataset}_{batchsize}_{dim}.pt')
    neg_relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransR/relations_{dataset}_{batchsize}_{dim}.pt')

    nrel = remb.shape[0]
    nnodes = eemb.shape[0]

    PROJ = relay.var('proj', shape=(nrel, dim, dim), dtype='float32')
    EMB = relay.var('emb', shape=(nnodes, 1, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, 1, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transR(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    tproj = tvm.nd.array(pemb)
    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(heads)
    tvt = tvm.nd.array(tails)
    tvr = tvm.nd.array(relations)

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
    
    # print(lib.lib.imported_modules[0].get_source())
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())
    

    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize * 64, ), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = transR_neg(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)
    tproj = tvm.nd.array(pemb)
    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(neg_heads)
    tvt = tvm.nd.array(neg_tails)
    tvr = tvm.nd.array(neg_relations)

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

    print(pos_score.shape, neg_score.shape)

def RESCAL_tvm(batchsize, dim):
    eemb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/RESCAL/eemb_{dataset}_{batchsize}_{dim}.pt')
    remb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/RESCAL/remb_{dataset}_{batchsize}_{dim}.pt')
    
    heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/RESCAL/heads_{dataset}_{batchsize}_{dim}.pt')
    tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/RESCAL/tails_{dataset}_{batchsize}_{dim}.pt')
    relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/RESCAL/relations_{dataset}_{batchsize}_{dim}.pt')

    neg_heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/RESCAL/heads_{dataset}_{batchsize}_{dim}.pt')
    neg_tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/RESCAL/tails_{dataset}_{batchsize}_{dim}.pt')
    neg_relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/RESCAL/relations_{dataset}_{batchsize}_{dim}.pt')

    nrel = remb.shape[0]
    nnodes = eemb.shape[0]
    EMB = relay.var('emb', shape=(nnodes, 1, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = RESCAL(EMB, RELEMB, VH, VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(heads)
    tvt = tvm.nd.array(tails)
    tvr = tvm.nd.array(relations)

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

    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize*64,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = RESCAL_neg(EMB, RELEMB, VH, VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(neg_heads)
    tvt = tvm.nd.array(neg_tails)
    tvr = tvm.nd.array(neg_relations)

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

    print(pos_score.shape, neg_score.shape)

def TransH_tvm(batchsize, dim):
    eemb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransH/eemb_{dataset}_{batchsize}_{dim}.pt')
    remb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransH/remb_{dataset}_{batchsize}_{dim}.pt')
    pemb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransH/pemb_{dataset}_{batchsize}_{dim}.pt')
    
    heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransH/heads_{dataset}_{batchsize}_{dim}.pt')
    tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransH/tails_{dataset}_{batchsize}_{dim}.pt')
    relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransH/relations_{dataset}_{batchsize}_{dim}.pt')

    neg_heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransH/heads_{dataset}_{batchsize}_{dim}.pt')
    neg_tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransH/tails_{dataset}_{batchsize}_{dim}.pt')
    neg_relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransH/relations_{dataset}_{batchsize}_{dim}.pt')

    nrel = remb.shape[0]
    nnodes = eemb.shape[0]
    PROJ = relay.var('proj', shape=(nrel, dim), dtype='float32')
    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = transH(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)
    tproj = tvm.nd.array(pemb)
    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(heads)
    tvt = tvm.nd.array(tails)
    tvr = tvm.nd.array(relations)

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
    print(lib.lib.imported_modules[0].get_source())
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())


    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize * 64, ), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = transH_neg(PROJ, EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)
    tproj = tvm.nd.array(pemb)
    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(neg_heads)
    tvt = tvm.nd.array(neg_tails)
    tvr = tvm.nd.array(neg_relations)

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

    print(pos_score.shape, neg_score.shape)

def TransF_tvm(batchsize, dim):
    eemb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransF/eemb_{dataset}_{batchsize}_{dim}.pt')
    remb = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransF/remb_{dataset}_{batchsize}_{dim}.pt')
    
    heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransF/heads_{dataset}_{batchsize}_{dim}.pt')
    tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransF/tails_{dataset}_{batchsize}_{dim}.pt')
    relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/TransF/relations_{dataset}_{batchsize}_{dim}.pt')

    neg_heads = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransF/heads_{dataset}_{batchsize}_{dim}.pt')
    neg_tails = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransF/tails_{dataset}_{batchsize}_{dim}.pt')
    neg_relations = torch.load(f'/data/not_backed_up/lihhu/KGEembeddings/neg/TransF/relations_{dataset}_{batchsize}_{dim}.pt')

    nrel = remb.shape[0]
    nnodes = eemb.shape[0]

    EMB = relay.var('emb', shape=(nnodes, dim), dtype='float32')
    RELEMB = relay.var('relemb', shape=(nrel, dim), dtype='float32')
    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')
    act = transF(EMB, RELEMB, VH,VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(heads)
    tvt = tvm.nd.array(tails)
    tvr = tvm.nd.array(relations)

    params = {}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=params)
    m = graph_executor.GraphModule(lib['default'](dev))
    m.set_input('emb', temb)
    m.set_input('relemb', trelemb)
    m.set_input("vh", tvh)
    m.set_input("vt", tvt)
    m.set_input("vr", tvr)
    print(lib.lib.imported_modules[0].get_source())
    print(m.benchmark(dev, number=1, repeat=1))
    pos_score = torch.from_numpy(m.get_output(0).asnumpy())


    VH = relay.var('vh', shape=(batchsize,), dtype='int64')
    VT = relay.var('vt', shape=(batchsize,64,), dtype='int64')
    VR = relay.var('vr', shape=(batchsize,), dtype='int64')

    act = TransF_neg(EMB, RELEMB, VH, VT, VR)
    func = relay.Function(relay.analysis.free_vars(act), act)
    mod = tvm.IRModule.from_expr(func)

    temb = tvm.nd.array(eemb)
    trelemb = tvm.nd.array(remb)
    tvh = tvm.nd.array(neg_heads)
    tvt = tvm.nd.array(neg_tails)
    tvr = tvm.nd.array(neg_relations)

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

    print(pos_score.shape, neg_score.shape)

    nhh = eemb[neg_heads]
    ntt = eemb[neg_tails]
    nrr = remb[neg_relations]
    neg_score2 = 2 * torch.einsum('ab,acb->ac', nhh, ntt) + torch.einsum('ab,acb->ac', nrr, (ntt - nhh.unsqueeze(1)))
    print(1000*neg_score[0], 1000*neg_score2[0])
    print(torch.sum(torch.abs(neg_score-neg_score2)))


if __name__ == '__main__':
    if args.model == 'TransE_l1':
        TransE_tvm(args.batchsize, args.dim)
    elif args.model == 'TransR':
        TransR_tvm(args.batchsize, args.dim)
    elif args.model == 'TransH':
        TransH_tvm(args.batchsize, args.dim)
    elif args.model == 'TransF':
        TransF_tvm(args.batchsize, args.dim)
    elif args.model == 'RESCAL':
        RESCAL_tvm(args.batchsize, args.dim)
    