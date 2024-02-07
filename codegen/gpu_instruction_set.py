from codegen import *
from ir import *
from asg import *
from asg2ir import gen_ir

class GridDimy(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class GridDimx(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class BlockIdy(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class BlockIdx(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class BlockDimy(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class BlockDimx(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class ThreadIdy(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class ThreadIdx(DObject):
    def __init__(self, dtype='int', size=[]):
        super().__init__(dtype, size)
        # self.dtype = 'int'
        # self.size = []

class SyncThreads(IR):
    def __init__(self):
        super().__init__()

class SyncWarps(IR):
    def __init__(self):
        super().__init__()

class ShuffleDown(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class ShuffleUp(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class ShuffleXor(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class SaveAtThread(IR):
    def __init__(self, src, dst, threadid):
        super().__init__()
        self.src = src
        self.dst = dst
        self.threadid = threadid

class BroadCast(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class Shared(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject


def ir2gpu(ir):
    match ir.__class__.__name__:
        case 'GridDimx':
            return 'gridDimx.x'
        case 'GridDimy':
            return 'gridDimx.y'
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