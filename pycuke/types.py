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