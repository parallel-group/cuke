from __future__ import annotations
import inspect
from . import helpers

MIN_INT = -2147483648
MAX_INT = 2147483647

arith_op = {'add': '+', 'sub': '-', 'mul': '*', 'floordiv': '/', 'truediv': '/', 'mod': '%'}
math_op = ['round', 'abs', 'nbits']
cmp_op = ['bigger', 'smaller']
func_op = ['apply', 'reduce', 'aggr']
other_op = ['setval', 'einsum', 'index', 'inline', 'size', 'norm', 'view']

binary_elw = list(arith_op.keys()) + cmp_op
unary_elw = math_op
elementwise_op = binary_elw + unary_elw

int_types = ['int', 'int32_t', 'int64_t']
float_types = ['float', 'double']


def new_op(func):
    def wrapper_func(*args, **kwargs):
        _res = func(*args, **kwargs)
        _res.attr['op_name'] = func.__name__
        return _res
    return wrapper_func

def bigger(x, y):
    return TensorOp('bigger', x, y)


def smaller(x, y):
    return TensorOp('smaller', x, y)


# By default, the output of func should have the same size for any input, but they can have different sizes in the first dim if out_ofss is provided
def apply(func, data: (list, tuple), axes=None, out_ofs=None, cond=None):
    assert callable(func)
    nparam = len(inspect.signature(func).parameters)
    assert len(data) == nparam
    if axes == None:
        axes = []
    while (len(axes) < nparam):
        axes.append(Const(0, 'int'))
    return TensorOp('apply', func, *data, *axes, out_ofs, cond)


def setval(val, name='', dest=None):
    if isinstance(val, Tensor):
        if dest == None:
            if len(val.ref_size) > 0:
                if name == '':
                    res = Tensor(val.ref_size, dtype=val.dtype)
                else:
                    res = Tensor(val.ref_size, name=name, dtype=val.dtype)
            else:
                if name == '':
                    res = Var(dtype=val.dtype)
                else:
                    res = Var(name=name, dtype=val.dtype)
        else:
            res = dest
    else:
        if type(val) == int:
            if dest == None:
                if name == '':
                    res = Var(dtype='int')
                else:
                    res = Var(name=name, dtype='int')
            else:
                res = dest
        elif type(val) == float:
            if dest == None:
                if name == '':
                    res = Var(dtype='float')
                else:
                    res = Var(name=name, dtype='float')
            else:
                res = dest

    res.attr['is_arg'] = False
    return TensorOp('setval', res, val)

def inline(src, output=[], inputs=[]):
    return TensorOp('inline', src, len(output), *[*output, *inputs])

def einsum(exp: str, tensor1, tensor2):
    return TensorOp('einsum', tensor1, tensor2, exp)

class ASTNode:
    nuniq = 0

    def __init__(self):
        self.decl = []
        self.compute = []
        self.output_order = []
        self.input_orders = []
        self.eval = None
        self.ref_by = []
        self.id = ASTNode.nuniq
        ASTNode.nuniq += 1
        self.attr = {}


class Tensor(ASTNode):
    def __init__(self, size: list | tuple, dtype='float', name=None):
        super().__init__()
        self.ref_size = []
        for s in size:
            if helpers.is_int_var(s):
                self.ref_size.append(s)
            elif type(s) == int:
                self.ref_size.append(Const(s, 'int'))
            else:
                raise TypeError('tensor dimensions must be int or a scalar int variable')
        self.dtype = dtype
        self.name = name
        self.attr['is_arg'] = True

    def __sub__(self, other):
        return TensorOp('sub', self, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __add__(self, other):
        return TensorOp('add', self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return TensorOp('mul', self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return TensorOp('truediv', self, other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __floordiv__(self, other):
        return TensorOp('floordiv', self, other)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __matmul__(self, other):
        return TensorOp('einsum', self, other, 'ij,jk->ik')

    def __getitem__(self, idx):
        assert isinstance(idx, (int, slice, Tensor, tuple))
        return TensorOp('index', self, idx)


    def apply(self, func, axis=0, out_ofs=None, cond=None):
        assert callable(func)
        return TensorOp('apply', func, self, axis, out_ofs, cond)

    def reduce(self, func, init, axis=0):
        if callable(func) and callable(init):
            return TensorOp('reduce', self, func, init, axis)
        else:
            raise TypeError('reduce must use a callable function')

    @new_op
    def sum(self, axis=0):
        s1 = ''
        rs = ''
        for i in range(len(self._size())):
            s1 += chr(ord('i') + i)
            if i != axis:
                rs += chr(ord('i') + i)
        return einsum(f'{s1},->{rs}', self, None)

    def max(self, axis=0):
        func = lambda x, y: bigger(x, y)
        init = lambda: setval(MIN_INT)
        return self.reduce(func, init, axis)

    def min(self, axis=0):
        func = lambda x, y: smaller(x, y)
        init = lambda: setval(MAX_INT)
        return self.reduce(func, init, axis)

    def aggr(self, func, init, indices, axis=0, size=None):
        if callable(func) and callable(init):
            op = TensorOp('aggr', self, func, init, indices, axis, size)
            return op
        else:
            raise TypeError('aggr must use a callable function')

    def aggr_sum(self, indices, axis=0, size=None):
        func = lambda x, y: x + y
        init = lambda: setval(0)
        return self.aggr(func, init, indices, axis, size)

    def aggr_max(self, indices, axis=0, size=None):
        func = lambda x, y: bigger(x, y)
        init = lambda: setval(MIN_INT)
        return self.aggr(func, init, indices, axis, size)

    def aggr_min(self, indices, axis=0, size=None):
        func = lambda x, y: smaller(x, y)
        init = lambda: setval(MAX_INT)
        return self.aggr(func, init, indices, axis, size)

    def prefix_sum(self, axis=0, inclusive=True):
        assert type(axis) == int or (type(axis) == Const and axis.dtype in int_types), 'axis must be an integer or a Const'
        if type(axis) == Const:
            axis = axis.val

        assert len(self.ref_size) > 0, 'input must have at least one dimension'
        data = self
        size = []
        if not inclusive:
            size[0] = self.ref_size[axis] + 1

        for i in range(len(self.ref_size)):
            if axis != i:
                size.append(self.ref_size[i])

        res = Tensor(size, dtype=self.dtype)

        for i in range(axis):
            data = data[:]
            res = res[:]

        if inclusive:
            return setval(data[:] + res[-1:size[axis] - 1], dest=res)
        else:
            return setval(data[-1:data._size()[axis]] + res[-1:size[axis] - 1], dest=res)

    def _size(self):
        return self.ref_size

    def size(self, axis):
        return TensorOp('size', self, axis)

    def round(self):
        return TensorOp('round', self)

    def abs(self):
        return TensorOp('abs', self)

    def nbits(self):
        return TensorOp('nbits', self)

    def view(self, sizes, dims):
        return TensorOp('view', self, sizes, dims)


class Var(Tensor):
    def __init__(self, dtype='int', name=None):
        super().__init__([], dtype, name)


# const is var without name
class Const(Var):
    def __init__(self, val, dtype):
        super().__init__(dtype)
        # slice is considered constant because once the slice is created its start, stop, step cannot be reassigned
        # however, start, stop, step themselves can be variables
        if dtype == 'slice':
            if type(val.start) == int:
                start = Const(val.start, 'int')
            else:
                start = val.start
            if type(val.stop) == int:
                stop = Const(val.stop, 'int')
            else:
                stop = val.stop
            if type(val.step) == int:
                step = Const(val.step, 'int')
            else:
                step = val.step
            assert helpers.is_int_var(start)
            assert helpers.is_int_var(stop)
            assert helpers.is_int_var(step)
            self.val = slice(start, stop, step)
        else:
            self.val = val



class TensorOp(Tensor):
    Types = func_op + list(arith_op.keys()) + math_op + cmp_op + other_op

    def __init__(self, op_type, *operators):
        assert op_type in TensorOp.Types
        self.op_type = op_type

        # TODO: infer result data type
        self.operators = []
        for opr in operators:
            self.operators.append(opr)
            if isinstance(opr, ASTNode) and (not op_type in ('setval', 'inline')):
                opr.ref_by.append(self)

        if op_type in arith_op or op_type in cmp_op:
            # TODO: infer dtype
            dtype = operators[0].dtype
            if type(self.operators[0]) == int:
                # assert(dtype in int_types)
                self.operators[0] = Const(self.operators[0], dtype)
            elif type(operators[0]) == float:
                # assert(dtype in float_types)
                self.operators[0] = Const(self.operators[0], dtype)
            if type(self.operators[1]) == int:
                # assert(dtype in int_types)
                self.operators[1] = Const(self.operators[1], dtype)
            elif type(operators[1]) == float:
                # assert(dtype in float_types)
                self.operators[1] = Const(self.operators[1], dtype)
            assert helpers.prefix_match_size(self.operators[0]._size(), self.operators[1]._size())
            if (len(self.operators[0]._size()) > len(self.operators[1]._size())):
                ref_size = self.operators[0]._size()
            else:
                ref_size = self.operators[1]._size()

        elif op_type == 'einsum':
            dtype = operators[0].dtype
            exp = self.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            op1_size = self.operators[0]._size()
            if self.operators[1] != None:
                op2_size = self.operators[1]._size()
            else:
                op2_size = []
            ref_size = []
            for i in output:
                pos1 = input1.find(i)
                if pos1 >= 0:
                    ref_size.append(op1_size[pos1])
                else:
                    pos2 = input2.find(i)
                    if pos2 >= 0:
                        ref_size.append(op2_size[pos2])
                    else:
                        raise IndexError('index not found!')

        elif op_type == 'view':
            dtype = operators[0].dtype
            sizes = []
            for s in operators[1]:
                if (type(s) == Const and s.dtype in int_types) or helpers.is_int_var(s):
                    sizes.append(s)
                elif type(s) == int:
                    sizes.append(Const(s, 'int'))
                else:
                    raise TypeError('tensor dimensions must be int or a scalar int variable')
            dims = []
            # view dims must be int literal or const
            for s in operators[2]:
                if type(s) == Const and s.dtype in int_types:
                    dims.append(s)
                elif type(s) == int:
                    dims.append(Const(s, 'int'))
                elif type(s) in (list, tuple):
                    d = []
                    for ss in s:
                        if type(ss) == Const and ss.dtype in int_types:
                            d.append(ss)
                        elif type(ss) == int:
                            d.append(Const(ss, 'int'))
                    dims.append(d)
                else:
                    raise TypeError('tensor dimensions must be int or a scalar int variable')
            # TODO: check if sizes is consistent with ref_size of the original tensor
            ref_size = sizes
            self.operators.pop()
            self.operators[1] = dims

        elif op_type == 'index':
            dtype = operators[0].dtype
            
            if not type(operators[1]) in (list, tuple):
                self.operators[1] = [operators[1]]
            ref_size = self.operators[0]._size()[len(self.operators[1]):]

            new_size = []
            new_idx = []
            size_dtype = operators[0]._size()[0].dtype
            for i in range(len(self.operators[1])):
                idx = self.operators[1][i]
                if type(idx) == slice:
                    start = idx.start
                    if start == None:
                        start = Const(0, size_dtype)
                    elif type(start) == int:
                        start = Const(start, size_dtype)
                    stop = idx.stop
                    if stop == None:
                        stop = self.operators[0].ref_size[i]
                    elif type(stop) == int:
                        stop = Const(stop, size_dtype)
                    step = idx.step
                    if step == None:
                        step = Const(1, size_dtype)
                    elif type(step) == int:
                        step = Const(step, size_dtype)

                    idx = Const(slice(start, stop, step), 'slice')

                    if step.val == 1:
                        if type(start) == Const and start.val==0:
                            csize = [helpers.eval_const_expr(stop)]
                        else:
                            csize = [helpers.eval_const_expr(stop - start)]
                    else:
                        csize = [helpers.eval_const_expr(stop - start) // step]
                elif helpers.is_1dint_tensor(idx):
                    csize = [idx.ref_size[0]]
                elif helpers.is_int_var(idx) or type(idx) == int:
                    csize = []
                    if type(idx) == int:
                        idx = Const(idx, size_dtype)
                else:
                    raise TypeError('index data type error!')

                new_size.extend(csize)
                new_idx.append(idx)

            ref_size[0:0] = new_size
            self.operators.pop()
            self.operators.extend(new_idx)

        elif op_type == 'apply':
            func = self.operators[0]
            self.nparams = len(inspect.signature(func).parameters)

            # check the axes of the n parameters, an axis must be an int or a int Const
            for i in range(self.nparams):
                axis = operators[1 + self.nparams + i]
                if type(axis) == int:
                    self.operators[1 + self.nparams + i] = Const(axis, 'int')
                else:
                    assert type(axis) == Const and axis.dtype in int_types, 'invalid axis'

            data = []
            # this is the primary axis size, i.e., number of loop iterations
            axis_size = self.operators[1].ref_size[self.operators[1 + self.nparams].val]
            for i in range(1, 1 + self.nparams):
                data_size = self.operators[i].ref_size
                axis = self.operators[self.nparams + i].val
                # every input item should have the same size as the primary axis size
                assert helpers.has_same_value(axis_size, data_size[axis])
                item_size = data_size[:axis] + data_size[axis + 1:]
                if (len(item_size) > 0):
                    item = Tensor(item_size, self.operators[i].dtype)
                else:
                    item = Var(self.operators[i].dtype)
                item.attr['is_arg'] = False
                data.append(item)

            # call function on an item of the input data
            ret = self.operators[0](*data)
            dtype = ret.dtype
            ret.ref_by.append(self)

            # handle output with offsets
            out_ofs = self.operators[1 + 2 * self.nparams]
            if out_ofs == None:
                ref_size = [axis_size] + ret.ref_size
            else:
                ref_size = [out_ofs[axis_size]] + ret.ref_size[1:]

            self.operators.extend(data)
            self.operators.append(ret)

            # handle conditional apply
            cond = self.operators[2 + 2 * self.nparams]
            if cond != None:
                assert helpers.is_1d_tensor(cond)
                assert helpers.has_same_value(axis_size, cond.ref_size[0])
                self.counter = setval(0)
                self.operators.append(self.counter)
                # set dynamic_size attribute for memory allocation optimization
                ref_size[0].attr['dynamic_size'] = True
            else:
                self.operators.append(None)

        elif op_type == 'reduce':
            assert type(self.operators[3]) == int
            axis = self.operators[3]
            self.operators[3] = Const(axis, 'int')
            ref_size = self.operators[0]._size()[:axis] + self.operators[0]._size()[axis + 1:]
            dtype = self.operators[0].dtype
            if (len(ref_size) > 0):
                item1 = Tensor(ref_size, self.operators[0].dtype)
                item2 = Tensor(ref_size, self.operators[0].dtype)
            else:
                item1 = Var(self.operators[0].dtype)
                item2 = Var(self.operators[0].dtype)
            item1.attr['is_arg'] = False
            item2.attr['is_arg'] = False

            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))

        elif op_type == 'aggr':
            dtype = operators[0].dtype
            assert helpers.is_1dint_tensor(self.operators[3])
            assert type(self.operators[4]) == int
            axis = self.operators[4]
            self.operators[4] = Const(axis, 'int')
            if self.operators[5] == None:
                self.operators[5] = self.operators[3].ref_size[0]
            else:
                assert helpers.is_int_var(self.operators[5])
                if type(self.operators[5]) == int:
                    self.operators[5] = Const(self.operators[5], 'int')
            ref_size = [self.operators[5]] + self.operators[0]._size()[:axis] + self.operators[0]._size()[axis + 1:]
            if (len(ref_size) > 1):
                item1 = Tensor(ref_size[1:], self.operators[0].dtype)
                item2 = Tensor(ref_size[1:], self.operators[0].dtype)
            else:
                item1 = Var(self.operators[0].dtype)
                item2 = Var(self.operators[0].dtype)
            item1.attr['is_arg'] = False
            item2.attr['is_arg'] = False
            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))

        elif op_type in math_op:
            dtype = self.operators[0].dtype
            ref_size = self.operators[0]._size()
            if op_type == 'round':
                dtype = 'int'
            elif op_type == 'abs':
                dtype = self.operators[0].dtype

        elif op_type == 'setval':
            dtype = self.operators[0].dtype
            ref_size = self.operators[0]._size()
            if (helpers.is_scalar(self.operators[1])):
                if type(self.operators[1]) == int:
                    assert(dtype in int_types)
                    self.operators[1] = Const(self.operators[1], dtype)
                elif type(self.operators[1]) == float:
                    assert(dtype in float_types)
                    self.operators[1] = Const(self.operators[1], dtype)

            else:
                assert self.operators[0].dtype == self.operators[1].dtype

        elif op_type == 'inline':
            src = self.operators[0]
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            num_output = self.operators[1]
            
            dtype = self.operators[2][1].dtype
            ref_size = self.operators[2][1]._size()

            keyvalue = []
            for op in self.operators[2:]:
                keyvalue.append(op[0])
                keyvalue.append(op[1])
            self.operators.clear()
            self.operators.append(src)
            self.operators.append(num_output)
            self.operators.extend(keyvalue)

        elif op_type == 'size':
            dtype = self.operators[0]._size()[0].dtype
            ref_size = []
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            axis = self.operators[1].val
            assert axis < len(self.operators[0]._size())


        super().__init__(ref_size, dtype, name = f'{op_type}_' + '_'.join([op.name if (hasattr(op, 'name') and op.name != None) else '' for op in self.operators]))

        # call the init function for reduce and aggr
        if self.op_type in ('reduce', 'aggr'):
            self.operators[2] = self.operators[2](self)

        self.input_orders = [[] for o in self.operators]





