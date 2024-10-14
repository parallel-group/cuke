class IR:
    def __init__(self):
        self.attr = {}


class Code(IR):
    def __init__(self, code: str, outputs:dict, inputs: dict):
        super().__init__()
        self.code = code
        self.outputs = outputs
        self.inputs = inputs


class DObject(IR):
    nobjects = 0

    def __init__(self, dtype: str, size: (list, tuple)):
        super().__init__()
        self.dobject_id = DObject.nobjects
        DObject.nobjects += 1
        self.dtype = dtype
        self.size = size

    def ref_size(self, axis):
        return self.size[axis]


class Expr(IR):
    def __init__(self, left, right, op: str, optional=None):
        super().__init__()
        self.left = left
        self.right = right
        self.optional = optional
        self.op = op
        self.size = self.left.size


class Assignment(IR):
    def __init__(self, lhs, rhs, op=None):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op


class Loop(IR):
    loop_id = 0
    parallel_types = ['naive', 'reduction', 'scan']

    def __init__(self, start, end, step, body: list, ptype='naive'):
        super().__init__()
        self.lid = Loop.loop_id
        Loop.loop_id += 1
        self.start = start
        self.end = end
        self.step = step
        self.body = body
        self.iterate = Scalar('int', f'_l{self.lid}')
        self.attr['ptype'] = ptype
        self.iterate.attr['loop'] = self


class FilterLoop(Loop):
    def __init__(self, start, end, step, cond, cond_body: list, body: list, ptype='naive'):
        super().__init__(start, end, step, body, ptype)
        self.cond = Indexing(cond, self.iterate)
        self.cond_body = cond_body


class Scalar(DObject):
    def __init__(self, dtype: str, name: str = None):
        super().__init__(dtype, [])
        self.__name__ = name if name else f's{self.dobject_id}'
        self.attr['is_arg'] = False

    def name(self):
        return self.__name__


class Literal(DObject):
    def __init__(self, val: (int, float), dtype: str):
        super().__init__(dtype, [])
        self.val = val


class Slice(IR):
    def __init__(self, start, stop, step):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = 'int'
        if self.step == 1 or (type(self.step) == Literal and self.step.val == 1):
            self.size = [Expr(self.stop, self.start, '-')]
        else:
            self.size = [Expr(Expr(self.stop, self.start, '-'), self.step, '/')]


class Ndarray(DObject):
    def __init__(self, dtype: str, size: tuple, name: str = None):
        super().__init__(dtype, size)
        self.__name__ = name if name else f'arr{self.dobject_id}'
        self.attr['is_arg'] = False

    def __getitem__(self, item):
        return f'{self.__name__}[{item}]'

    def name(self):
        return self.__name__


class Math(IR):
    def __init__(self, val, type):
        super().__init__()
        self.val = val
        self.type = type
        self.size = val.size


class Indexing(DObject):
    def __init__(self, dobject, idx):
        assert dobject != None and type(dobject) in (Slice, Ndarray, Indexing)
        assert idx != None and type(idx) in (Scalar, Literal, Indexing, Expr)
        self.dobject = dobject
        self.idx = idx

        if type(self.dobject) in (Ndarray, Slice):
            if type(idx) == Literal and idx.val == -1:
                # idx is unspecified, which means the Indexing is a range of indice stored in dobject, so the size of Indexing should the same as the dobject
                size = dobject.size[:]
                self.ref_point = 1
            else:
                # idx is a specific Scalar, Literal, or Indexing, in any case, the size of the Indexing operation should be as follows
                # ref_point should be the next dimension if the node is further Indexed
                size = idx.size + dobject.size[1:]
                self.ref_point = len(idx.size)
        else:
            # dobject is an Indexing
            size = dobject.size[:dobject.ref_point] + idx.size + dobject.size[dobject.ref_point + 1:]
            self.ref_point = dobject.ref_point + len(idx.size)

        super().__init__(dobject.dtype, size)

    def refresh_size(self):
        if isinstance(self, Indexing):
            if isinstance(self.dobject, Indexing):
                self.dobject.refresh_size()
            if isinstance(self.idx, Indexing):
                self.idx.refresh_size()
        if type(self.dobject) in (Ndarray, Slice):
            if type(self.idx) == Literal and self.idx.val == -1:
                # idx is unspecified, which means the Indexing is a range of indice stored in dobject, so the size of Indexing should the same as the dobject
                self.size = self.dobject.size[:]
                self.ref_point = 1
            else:
                # idx is a specific Scalar, Literal, or Indexing, in any case, the size of the Indexing operation should be as follows
                # ref_point should be the next dimension if the node is further Indexed
                self.size = self.idx.size + self.dobject.size[1:]
                self.ref_point = len(self.idx.size)
        else:
            # dobject is an Indexing
            self.size = self.dobject.size[:self.dobject.ref_point] + self.idx.size + self.dobject.size[self.dobject.ref_point + 1:]
            self.ref_point = self.dobject.ref_point + len(self.idx.size)

    def ref_size(self, axis):
        return self.size[axis]


class Decl(IR):
    def __init__(self, dobject: (Scalar, Ndarray)):
        super().__init__()
        self.dobject = dobject
