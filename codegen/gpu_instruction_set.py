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
    if isinstance(ir, GridDimx):
            return 'gridDimx.x'
    elif isinstance(ir, GridDimy):
            return 'gridDimx.y'
    elif isinstance(ir, BlockIdx):
            return 'blockIdx.x'
    elif isinstance(ir, BlockIdy):
            return 'blockIdx.y'
    elif isinstance(ir, BlockDimx):
            return 'blockDim.x'
    elif isinstance(ir, BlockDimy):
            return 'blockDim.y'
    elif isinstance(ir, ThreadIdy):
            return 'threadIdx.y'
    elif isinstance(ir, ThreadIdx):
            return 'threadIdx.x'
    elif isinstance(ir, SyncThreads):
            return '__syncthreads();\n'
    elif isinstance(ir, SyncWarps):
            return '__syncwarps();\n'