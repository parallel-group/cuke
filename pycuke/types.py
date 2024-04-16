MIN_INT = -2147483648
MAX_INT = 2147483647

arith_op = {'add': '+', 'sub': '-', 'mul': '*', 'floordiv': '/', 'truediv': '/', 'mod': '%'}
math_op = ['round', 'abs', 'nbits']
cmp_op = ['bigger', 'smaller']
func_op = ['apply', 'reduce', 'aggr']
other_op = ['setval', 'einsum', 'index', 'inline', 'size', 'norm', 'view', 'mklist']

binary_elw = list(arith_op.keys()) + cmp_op
unary_elw = math_op
elementwise_op = binary_elw + unary_elw

int_types = ['int', 'int32_t', 'int64_t']
float_types = ['float', 'double']

#https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h
type_map = {'int': 'kInt',
            'int32_t': 'kInt',
            'int64_t': 'kLong',
            'float': 'kFloat',
            'double': 'kDouble'}

def get_expr_type(ltype, rtype, op):
    # TODO: need more accurate type inference
    if op == 'truediv':
        return 'float'
    elif op == 'floordiv':
        return 'int'
    else:
        if ltype in int_types and rtype in int_types:
           res = ltype
        elif ltype in float_types:
            res = ltype
        else:
            res = rtype
        return res
