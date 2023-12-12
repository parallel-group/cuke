from batch.opt.ir import *
from codegen import *

def ir2gpu(ir):
    match ir.__class__.__name__:
        case 'BlockIdx':
            return 'blockIdx.x'
        case 'BlockIdy':
            return 'blockIdx.y'
        case 'BlockDimx':
            return 'blockDim.x'
        case 'BlockDimy':
            return 'blockDim.y'
        case 'ThreadIdy':
            return 'threadIdx.y'
        case 'ThreadIdx':
            return 'threadIdx.x'
        case 'SyncThreads':
            return '__syncthreads();\n'
        case 'SyncWarps':
            return '__syncwarps();\n'