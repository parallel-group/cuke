from .. import transform
from .. import asg, ir
from ..helpers import collect_ir, get_input_nodes, ASGTraversal, IRTraversal, replace_all_ref, ir_find_defs, ir_find_uses
import random
import string
import os
from .gpu_instruction_set import *

def to_string(stmt):
    if isinstance(stmt, Expr):
        if stmt.op in asg.arith_op.values():
            return f"({to_string(stmt.left)}" + f" {stmt.op} " + f"{to_string(stmt.right)})"
        elif stmt.op == 'bigger':
            return f"({to_string(stmt.left)} > {to_string(stmt.right)} ? ({to_string(stmt.left)}) : ({to_string(stmt.right)}))"
        elif stmt.op == 'smaller':
            return f"({to_string(stmt.left)} < {to_string(stmt.right)} ? ({to_string(stmt.left)}) : ({to_string(stmt.right)}))"
        elif stmt.op == 'ternary':
            return f"({to_string(stmt.left)} ? {to_string(stmt.right)} : {to_string(stmt.optional)})"
        else:
            return f"({to_string(stmt.left)}" + f" {stmt.op} " + f"{to_string(stmt.right)})"
    elif isinstance(stmt, Assignment):
            if stmt.op is None:
                return f"{to_string(stmt.lhs)} = {to_string(stmt.rhs)};\n"
            else:
                return f"{to_string(stmt.lhs)} {stmt.op}= {to_string(stmt.rhs)};\n"
    elif isinstance(stmt, Loop):
            code = ''
            if stmt.attr['ptype'] in ['naive', 'reduction'] and 'plevel' in stmt.attr and 'nprocs' in stmt.attr:
                code += f"for (int {to_string(stmt.iterate)} = {to_string(stmt.start)}; {to_string(stmt.iterate)} < {to_string(stmt.end)}; {to_string(stmt.iterate)} += {to_string(stmt.step)}) {{\n"
                for e in stmt.body:
                    if e:
                        code += to_string(e)
                code += "} \n"
                # if 'redu_eval' in stmt.attr:
                #     if isinstance(stmt.attr["redu_eval"], ir.Scalar):
                #         code += f'for (int off = blockDim.x/2; off > 0; off >>= 1) {{\n {to_string(stmt.attr["redu_eval"])} += __shfl_down_sync(0xffffffff, {to_string(stmt.attr["redu_eval"])}, off); \n}}\n{to_string(stmt.attr["redu_eval"])} = __shfl_sync(0xffffffff, {to_string(stmt.attr["redu_eval"])}, 0);\n__syncthreads();\n'
                #     elif isinstance(stmt.attr["redu_eval"], (ir.Ndarray, ir.Indexing)):
                #         if 'redu_res' in stmt.attr:
                #             if isinstance(stmt.attr['redu_res'], ir.Scalar):
                #                 code += f'for (int off = blockDim.x/2; off > 0; off >>= 1) {{\n {to_string(stmt.attr["redu_res"])} += __shfl_down_sync(0xffffffff, {to_string(stmt.attr["redu_res"])}, off); \n}}\nif(threadIdx.x==0) {to_string(stmt.attr["redu_eval"])} = {to_string(stmt.attr["redu_res"])};\n__syncthreads();\n'
                #             elif isinstance(stmt.attr['redu_res'], (ir.Ndarray, ir.Indexing)):
                #                 code += f'__syncthreads();\nif(threadIdx.x==0) {to_string(stmt.attr["redu_eval"])} = {to_string(stmt.attr["redu_res"])};\n'
            elif 'reduction' in stmt.attr and stmt.attr['reduction']:
                code += f'for (int {to_string(stmt.iterate)} = {to_string(stmt.start)}; {to_string(stmt.iterate)} > {to_string(stmt.end)}; {to_string(stmt.iterate)} >>= {to_string(stmt.step)}) {{\n' 
                for e in stmt.body:
                    if isinstance(e, Assignment):
                        code += f'{to_string(e.rhs)} += __shfl_down_sync(0xffffffff, {to_string(e.rhs)}, {to_string(stmt.iterate)});\n'
                    else:
                        code += f'to_string(e)'
                code += '}\n__syncthreads();\n'
                code += f'if(threadIdx.x == 0){{ {to_string(e.lhs)} = {to_string(e.rhs)}; }}\n'
            else:
                code += f"for (int {to_string(stmt.iterate)} = {to_string(stmt.start)}; {to_string(stmt.iterate)} < {to_string(stmt.end)}; {to_string(stmt.iterate)} += {to_string(stmt.step)}) {{\n"
                for e in stmt.body:
                    if e:
                        code += to_string(e)
                code += "} \n"
            if 'sync' in stmt.attr:
                code += '__syncthreads();\n'
            return code
    elif isinstance(stmt, (Scalar, Ndarray)):
            if 'offset' in stmt.attr:
                return f'{stmt.name()}-{to_string(stmt.attr["offset"])}'
            return stmt.name()
    elif isinstance(stmt, Literal):
            return str(stmt.val)
    elif isinstance(stmt, Indexing):
            if type(stmt.dobject) == Slice:
                if stmt.dobject.step == 1 or (type(stmt.dobject.step) == Literal and stmt.dobject.step.val == 1):
                    if stmt.dobject.start == 0 or (type(stmt.dobject.start) == Literal and stmt.dobject.start.val == 0):
                        return f'({to_string(stmt.idx)})'
                    else:
                        return f'(({to_string(stmt.dobject.start)})+({to_string(stmt.idx)}))'
                else:
                    if stmt.dobject.start == 0 or (type(stmt.dobject.start) == Literal and stmt.dobject.start.val == 0):
                        return f'(({to_string(stmt.dobject.step)})*({to_string(stmt.idx)}))'
                    else:
                        return f'(({to_string(stmt.dobject.start)})+({to_string(stmt.dobject.step)})*({to_string(stmt.idx)}))'
            else:
                return f'{to_string(stmt.dobject)}[{to_string(stmt.idx)}]'
    elif isinstance(stmt, Decl):
            # variables are passed in as pytorch arguments
            if type(stmt.dobject) == Scalar:
                if not stmt.dobject.attr['is_arg']:
                    return f"{stmt.dobject.dtype} {stmt.dobject.name()};\n"
                else:
                    return ''
            elif type(stmt.dobject) == Ndarray:
                code = ''
                if not stmt.dobject.attr['is_arg']:
                    if 'mem_layer' in stmt.dobject.attr and stmt.dobject.attr['mem_layer'] == 'smem':
                        shape = ''
                        for s in stmt.dobject.size:
                            shape += f'[{s}]'
                        code = f'__shared__ {stmt.dobject.dtype} {stmt.dobject.name()}{shape};\n'
                    else:
                        code = f'torch::Tensor obj_{stmt.dobject.name()} = torch::empty({{{",".join([to_string(s) for s in stmt.dobject.size])}}}, torch::TensorOptions(torch::k{"Int" if stmt.dobject.dtype=="int" else "Float"}).device(torch::kCUDA));\n'
                return code
            else:
                return f'{to_string(stmt.dobject)}'
    elif isinstance(stmt, (ThreadIdx, ThreadIdy, BlockDimx, BlockDimy, BlockIdx, BlockIdy, GridDimx, GridDimy, SyncThreads, SyncWarps)):
            return ir2gpu(stmt)
    elif isinstance(stmt, Math):
            if isinstance(stmt.val, (list, tuple)):
                vals = f'{to_string(stmt.val[0])}'
                for i in range(1, len(stmt.val)):
                    vals += f', {to_string(stmt.val[i])}'
                return f"{stmt.type}({vals})"
            else:
                return f"{stmt.type}({to_string(stmt.val)})"
    elif isinstance(stmt, Code):
            code = stmt.code
            code = code.replace(stmt.output[0], to_string(stmt.output[1]))
            for kw in stmt.inputs:
                code = code.replace(kw, to_string(stmt.inputs[kw]))
            return code + '\n'
    elif isinstance(stmt, (list, tuple)):
            code = ''
            for s in stmt:
                code += to_string(s)
            return code
    else:
            return str(stmt)

def collect_cuda_ir(ast, stmt_cpu, stmt_gpu):
    def action_cpu(node, res):
        if type(node) == asg.Tensor:
            res.extend(node.decl)
    
    def action_gpu(node, res):
        if type(node) == asg.TensorOp:
            res.extend(node.decl)
            if not 'scope' in node.attr:
                res.extend(node.compute)

    t = ASGTraversal(action_cpu)
    stmt_cpu.extend(t(ast))

    t = ASGTraversal(action_gpu)
    stmt_gpu.extend(t(ast))

def cuda_spec(stmt, mapping):
    # replace_all_ref
    # todo: change assignment to for-loop
    def action(s, res):
        if s.__class__.__name__ == 'Loop':
            if s.attr['ptype'] in ['naive', 'reduction'] and 'plevel' in s.attr and 'nprocs' in s.attr:
                if s.attr['plevel'] == 0:
                    new_iter = ir.Expr(BlockIdx(), BlockDimy(), '*')
                    # replace_all_ref(s, s.iterate, new_iter)
                    s.start = new_iter
                    s.end = ir.Expr(new_iter, ir.Expr(s.end, mapping['block'], '/'), '+')
                    # s.step = GridDimx()
                    # mapping['block'] = s.attr['nprocs'][s.attr['plevel']][0]
                elif s.attr['plevel'] == 1:
                    s.start = ir.Expr(s.start, ThreadIdy(), '+')
                    s.step = BlockDimy()
                    # mapping['ty'] = s.attr['nprocs'][s.attr['plevel']][0]
                elif s.attr['plevel'] == 2:
                    s.start = ir.Expr(s.start, ThreadIdx(), '+')
                    s.step = BlockDimx()
                    # mapping['tx'] = s.attr['nprocs'][s.attr['plevel']][0]
                return [True, True, True, True, True]
            elif 'reduction' in s.attr and s.attr['reduction']:
                s.start = Expr(BlockDimx(), 2, '/')
                s.end = 0
            else:
                return [True, True, True, True, True]
        if isinstance(s, Assignment):
            def action(stmt, res):
                if stmt == s.lhs:
                    res.append(True)
                return [True, True, True, True, True]
            dfs = IRTraversal(action)(s)
        return [True, True, True, True, True]
    
    t = IRTraversal(action)
    res = t(stmt)
    # return res


def print_cuda(node):

    for p in transform.passes:
        node = p(node)
    
    # stmts = []
    # collect_ir(node, stmts)

    cpu_ir = []
    gpu_ir = []
    collect_cuda_ir(node, cpu_ir, gpu_ir)

    args = get_input_nodes(node)
    
    argscpu = ', '.join([f'torch::Tensor obj_{a}' if type(args[a]) == asg.Tensor else f'{args[a].dtype} {a}' for a in args])
    # argsptr = ', '.join([f'obj_{a}.data_ptr<{args[a].dtype}>()' if type(args[a]) == Tensor else f'{a}' for a in args])
    # ptrs = ', '.join([f'{args[a].dtype}* {a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])
    
    argsptr = ', '.join([f'obj_{a}.packed_accessor32<{args[a].dtype if args[a].dtype!="int" else "int64_t"}, {len(args[a].ref_size)}, torch::RestrictPtrTraits>()' if type(args[a]) == asg.Tensor else f'{a}' for a in args])
    ptrs = ', '.join([f'torch::PackedTensorAccessor32<{args[a].dtype if args[a].dtype!="int" else "int64_t"}, {len(args[a].ref_size)}, torch::RestrictPtrTraits> {a}' if type(args[a]) == asg.Tensor else f'{args[a].dtype} {a}' for a in args])
    # in cuda kernel:     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
    # host call cuda:     .packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    mapping = {'block':'(batch_size/16)', 'ty':16, 'tx':32}
    code = ''
    declare = ''
    cuda_declare = ''
    for d in gpu_ir:
        cuda_spec(d, mapping)
        if d:
            if type(d) == ir.Decl and type(d.dobject) == ir.Ndarray:
                if 'mem_layer' in d.dobject.attr and d.dobject.attr['mem_layer'] == 'smem':
                    cuda_declare += to_string(d)
                else:
                    declare += to_string(d)
                    argsptr += f', obj_{d.dobject.name()}.packed_accessor32<{d.dobject.dtype}, {len(d.dobject.size)}, torch::RestrictPtrTraits>()'
                    ptrs += f', torch::PackedTensorAccessor32<{d.dobject.dtype}, {len(d.dobject.size)}, torch::RestrictPtrTraits> {d.dobject.name()}'
            else:
                code += to_string(d)

    Return = ''
    if type(node.eval) == ir.Scalar:
        rtype = node.dtype
        Return = f'return {node.eval.name()};\n'
    elif type(node.eval) == ir.Ndarray:
        rtype = 'torch::Tensor'
        Return = f'return obj_{node.eval.name()};\n'
    else:
        rtype = 'void'

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_template.cu'), 'r') as f:
        c_code = f.read()
        for i in mapping:
            c_code = c_code.replace(i, str(mapping[i]))
        c_code = c_code.replace('RTYPE', rtype).replace('FNAME', node.name).replace('ARGS', argscpu).replace('CU_DE', cuda_declare).replace('CODE', code).replace('PTR_VARS', argsptr).replace('PTRS', ptrs).replace('DECL', declare).replace('RETURN', Return)
    return c_code