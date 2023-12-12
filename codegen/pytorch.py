import random
import string
from codegen.oob import lower_bound_padding
from codegen.tensorize import tensorize
from asg import *
from ir import *
import codegen


indent_width = 4

def to_string(ir, indent = 0):
    match ir.__class__.__name__:
        case 'Expr':
            if ir.op in arith_op.values():
                return f"({to_string(ir.left)}" + f" {ir.op} " + f"{to_string(ir.right)})"
            elif ir.op == 'bigger':
                return f"max({to_string(ir.left)}, {to_string(ir.right)})"
            elif ir.op == 'smaller':
                return f"min({to_string(ir.left)}, {to_string(ir.right)})"
        case 'Assignment':
            if ir.op is None:
                return ' ' * indent + f"{to_string(ir.lhs)} = {to_string(ir.rhs)}\n"
            else:
                return ' ' * indent + f"{to_string(ir.lhs)} {ir.op}= {to_string(ir.rhs)}\n"
        case 'Loop':
            code = ' ' * indent + f"for {to_string(ir.iterate)} in range({to_string(ir.start)}, {to_string(ir.end)}, {to_string(ir.step)}): \n"
            for e in ir.body:
                if e:
                    code += to_string(e, indent + indent_width)
            code += ' ' * indent + "\n"
            return code
        case 'Scalar' | 'Ndarray':
            return ir.name()
        case 'Literal':
            return str(ir.val)
        case 'Indexing':
            if type(ir.dobject) == Slice:
                return to_string(ir.dobject)
            else:
                return f'{to_string(ir.dobject)}[{to_string(ir.idx)}]'
        case 'Slice':
            if ir.step == 1 or (type(ir.step) == Literal and ir.step.val == 1):
                return f'{to_string(ir.start)}:{to_string(ir.stop)}'
            else:
                return f'{to_string(ir.start)}:{to_string(ir.stop)}:{to_string(ir.step)}'
        case 'Math':
            return f'{ir.type}({to_string(ir.val)})'
        case 'Decl':
            return ''
        case _:
            return str(ir)


def print_pytorch(asg):
    ir = []
    lower_bound_padding(asg)
    tensorize(asg)

    helpers.collect_ir(asg, ir)

    args = helpers.get_input_nodes(asg)
    args = ', '.join([f'{a}' for a in args])

    code = ''
    indent = indent_width
    for d in ir:
        if d:
            code += to_string(d, indent)

    code += ' ' * indent + f'return {asg.eval.name()}\n'

    with open('codegen/pytorch_template.txt', 'r') as f:
        t_code = f.read()
        t_code = t_code.replace('FNAME', asg.name[:24] + ''.join(random.choices(string.ascii_lowercase, k=8))).replace('ARGS', args).replace('CODE', code)
        return t_code



def test1():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (10, 20))
    ast = A + B
    code = codegen.pytorch.print_pytorch(ast._gen_ir())
    print(code)

def test2():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (10, ), dtype='int')
    ast = A[B] + 1
    code = codegen.pytorch.print_pytorch(ast._gen_ir())
    print(code)


def test3():
    input = Tensor('input', (10, 20))
    sampled_nodes = Tensor('sampled_nodes', (5, ), dtype='int')

    res = sampled_nodes.apply(lambda n: input[n])
    code = codegen.pytorch.print_pytorch(res._gen_ir())
    print(code)

def test4():
    ndevs = 4
    nnodes = 1000
    dim = 512
    batch_size = 32
    buffer_size = nnodes // ndevs
    buffers = Tensor('buffers', (ndevs, buffer_size, dim))
    sampled_nodes = Tensor('sampled_nodes', (batch_size, ), dtype='int')
    dev_ids = Tensor('dev_ids', (nnodes, ), dtype='int')
    buf_ofs = Tensor('buf_ofs', (nnodes, ), dtype='int')

    res = sampled_nodes.apply(lambda v: buffers[dev_ids[v]][buf_ofs[v]])
    code = codegen.pytorch.print_pytorch(res._gen_ir())
    print(code)

def test5():
    ndevs = 4
    nnodes = 1000
    dim = 512
    batch_size = 32
    buffer_size = nnodes // ndevs
    buffers = Tensor('buffers', (ndevs, buffer_size, dim))
    dev_ids = Tensor('dev_ids', (nnodes, ), dtype='int')
    buf_ofs = Tensor('buf_ofs', (nnodes, ), dtype='int')

    sampled_nodes = Tensor('sampled_nodes', (batch_size, ), dtype='int')

    # rowptr is obtained by sorting sampled_nodes according to their dev_ids
    rowptr = Tensor('rowptr', (ndevs+1, ), dtype='int')

    d = 0
    res = buffers[d][buf_ofs[sampled_nodes[rowptr[d]:rowptr[d+1]]]] + 0
    code = codegen.pytorch.print_pytorch(res._gen_ir())
    print(code)

if __name__ == "__main__":
    test5()
