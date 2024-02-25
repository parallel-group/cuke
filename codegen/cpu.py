import transform
from helpers import collect_ir, get_input_nodes
import random
import string
import os
import asg
import ir

#https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h
type_map = {'int': 'kInt', 
            'int32_t': 'kInt',
            'int64_t': 'kLong',
            'float': 'kFloat',
            'double': 'kDouble'}

def get_dtype(expr):
    if isinstance(expr, ir.Expr):
        return get_dtype(expr.left)
    elif isinstance(expr, ir.DObject):
        return expr.dtype
    else:
        return 'int64_t'

def to_string(stmt):
    match stmt.__class__.__name__:
        case 'Expr':
            if stmt.op in asg.arith_op.values():
                return f"({to_string(stmt.left)}" + f" {stmt.op} " + f"{to_string(stmt.right)})"
            elif stmt.op == 'bigger':
                return f"({to_string(stmt.left)} > {to_string(stmt.right)} ? ({to_string(stmt.left)}) : ({to_string(stmt.right)}))"
            elif stmt.op == 'smaller':
                return f"({to_string(stmt.left)} < {to_string(stmt.right)} ? ({to_string(stmt.left)}) : ({to_string(stmt.right)}))"
        case 'Assignment':
            if stmt.op is None:
                return f"{to_string(stmt.lhs)} = {to_string(stmt.rhs)};\n"
            else:
                return f"{to_string(stmt.lhs)} {stmt.op}= {to_string(stmt.rhs)};\n"
        case 'Loop':
            code = ''
            if stmt.attr['ptype'] in ['naive', 'reduction'] and 'plevel' in stmt.attr and 'nprocs' in stmt.attr:
                code += f"#pragma omp parallel for num_threads({stmt.attr['nprocs'][stmt.attr['plevel']][0]})\n"
            code += f"for ({get_dtype(stmt.end)} {to_string(stmt.iterate)} = {to_string(stmt.start)}; {to_string(stmt.iterate)} < {to_string(stmt.end)}; {to_string(stmt.iterate)} += {to_string(stmt.step)}) {{\n"
            for e in stmt.body:
                if e:
                    code += to_string(e)
            code += "} \n"
            return code
        case 'FilterLoop':
            code = ''
            if stmt.attr['ptype'] == 'naive' and 'plevel' in stmt.attr and 'nprocs' in stmt.attr:
                code += f"#pragma omp parallel for num_threads({stmt.attr['nprocs'][stmt.attr['plevel']][0]})\n"
            code += f"for ({get_dtype(stmt.end)} {to_string(stmt.iterate)} = {to_string(stmt.start)}; {to_string(stmt.iterate)} < {to_string(stmt.end)}; {to_string(stmt.iterate)} += {to_string(stmt.step)}) {{\n"
            for e in stmt.cond_body:
                if e:
                    code += to_string(e)
            code += f"if ({to_string(stmt.cond)}) {{\n"
            for e in stmt.body:
                if e:
                    code += to_string(e)
            code += "} \n"
            code += "} \n"
            return code
        case 'Scalar' | 'Ndarray':
            return stmt.name()
        case 'Literal':
            return str(stmt.val)
        case 'Indexing':
            if type(stmt.dobject) == ir.Slice:
                if stmt.dobject.step == 1 or (type(stmt.dobject.step) == ir.Literal and stmt.dobject.step.val == 1):
                    if stmt.dobject.start == 0 or (type(stmt.dobject.start) == ir.Literal and stmt.dobject.start.val == 0):
                        return f'({to_string(stmt.idx)})'
                    else:
                        return f'(({to_string(stmt.dobject.start)})+({to_string(stmt.idx)}))'
                else:
                    if stmt.dobject.start == 0 or (type(stmt.dobject.start) == ir.Literal and stmt.dobject.start.val == 0):
                        return f'(({to_string(stmt.dobject.step)})*({to_string(stmt.idx)}))'
                    else:
                        return f'(({to_string(stmt.dobject.start)})+({to_string(stmt.dobject.step)})*({to_string(stmt.idx)}))'
            else:
                return f'{to_string(stmt.dobject)}[{to_string(stmt.idx)}]'
        case 'Decl':
            # variables are passed in as pytorch arguments
            if type(stmt.dobject) == ir.Scalar:
                if not stmt.dobject.attr['is_arg']:
                    return f"{stmt.dobject.dtype} {stmt.dobject.name()};\n"
                else:
                    return ''
            elif type(stmt.dobject) == ir.Ndarray:
                code = ''
                if not stmt.dobject.attr['is_arg']:
                    code = f'torch::Tensor obj_{stmt.dobject.name()} = torch::empty({{{",".join([to_string(s) for s in stmt.dobject.size])}}}, at::{type_map[stmt.dobject.dtype]});\n'
                code += f'auto {stmt.dobject.name()} = obj_{stmt.dobject.name()}.accessor<{stmt.dobject.dtype}, {len(stmt.dobject.size)}>();\n'
                return code
        case 'Math':
            return f"{stmt.type}({to_string(stmt.val)})"
        case 'Code':
            code = stmt.code
            for kw in stmt.outputs:
                code = code.replace(kw, to_string(stmt.outputs[kw]))
            for kw in stmt.inputs:
                code = code.replace(kw, to_string(stmt.inputs[kw]))
            return code + '\n'
        case 'list' | 'tuple':
            code = ''
            for s in stmt:
                code += to_string(s)
            return code
        case _:
            return str(stmt)


def print_cpp(node):

    for p in transform.passes:
        node = p(node)

    stmts = []
    collect_ir(node, stmts)

    # fix dynamic size allocation
    for j in range(len(stmts)):
        d = stmts[j]
        if type(d) == ir.Decl:
            for i in range(len(d.dobject.size)):
                if type(d.dobject.size[i]) == ir.Scalar:
                    if 'dynamic_size' in d.dobject.size[i].attr:
                        d.dobject.size[i] = ir.Literal(4096, 'int')
                        # TODO: remove the decl of the variable if it is not used elsewhere

    args = get_input_nodes(node)
    args = ', '.join([f'torch::Tensor obj_{a}' if type(args[a]) == asg.Tensor else f'{args[a].dtype} {a}' for a in args])

    code = ''
    for d in stmts:
        if d:
            code += to_string(d)


    if type(node.eval) == ir.Scalar:
        rtype = node.dtype
        code += f'return {node.eval.name()};\n'
    elif type(node.eval) == ir.Ndarray:
        rtype = 'torch::Tensor'
        code += f'return obj_{node.eval.name()};\n'
    else:
        rtype = 'void'
        # raise TypeError('wrong output type', node.eval)

    with open(f'{os.path.dirname(__file__)}/cpp_template.txt', 'r') as f:
        c_code = f.read()
        c_code = c_code.replace('RTYPE', rtype).replace('FNAME', node.name[:24] + ''.join(random.choices(string.ascii_lowercase, k=8))).replace('ARGS', args).replace('CODE', code)
    return c_code
