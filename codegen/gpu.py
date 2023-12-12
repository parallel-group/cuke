import helpers
from codegen.gpu_instructionsets import *

def to_string(ir):
    match ir.__class__.__name__:
        case 'Expr':
            return f"({to_string(ir.left)}" + f" {ir.op} " + f"{to_string(ir.right)})"
        case 'Assignment':
            if ir.op is None:
                return f"{to_string(ir.lhs)} = {to_string(ir.rhs)};\n"
            else:
                return f"{to_string(ir.lhs)} {ir.op}= {to_string(ir.rhs)};\n"
        case 'Loop':
            code = f"for (int {to_string(ir.iterate)} = {to_string(ir.start)}; {to_string(ir.iterate)} < {to_string(ir.end)}; {to_string(ir.iterate)} += {to_string(ir.step)}) {{\n"
            # print(ir, ir.body, to_string(ir.body))
            for e in ir.body:
                if e:
                    code += to_string(e)
            code += "} \n"
            return code
        case 'Scalar' | 'Ndarray' | 'Ref':
            return ir.name()
        case 'Literal':
            return str(ir.val)
        case 'Indexing':
            if type(ir.dobject) == Slice:
                return f'(({to_string(ir.dobject.start)})+({to_string(ir.dobject.step)})*({to_string(ir.idx)}))'
            # elif type(ir.dobject) == Pointer:
            #     code = f'{to_string(ir.dobject)}['
            #     for i in range(len(ir.dobject.dims)):
            #         code += f'{to_string(ir.idx)}*{to_string(ir.dobject.dims[i])}'
            #         if i<len(ir.dobject.dims)-1:
            #             code += ' + '
            #     # for item in ir.dobject.dims:
            #     #     code += f'{to_string(ir.idx)}*{to_string(item)}'
                    
            #     code += ']'
            #     return code
            else:
                return f'{to_string(ir.dobject)}[{to_string(ir.idx)}]'
        case 'Decl':
            # variables are passed in as pytorch arguments
            
            if type(ir.dobject) == Scalar:
                if not ir.dobject.is_arg:
                    return f"{ir.dobject.dtype} {ir.dobject.name()};\n"
                else:
                    return ''
            elif type(ir.dobject) == Ndarray:
                code = ''
                if not ir.dobject.is_arg:
                    code = f'torch::Tensor obj_{ir.dobject.name()} = torch::empty({{{",".join([to_string(s) for s in ir.dobject.size])}}}, torch::TensorOptions(torch::k{"Int" if ir.dobject.dtype=="int" else "Float"}).device(torch::kCUDA));\n'
                return code
            elif type(ir.dobject) == Shared:
                shape = ''
                for i in range(len(ir.dobject.dobject.size)):
                    shape += f'[{to_string(ir.dobject.dobject.size[i])}]'
                return f'__shared__ {ir.dobject.dobject.dtype} {ir.dobject.dobject.name()}{shape};\n'
            elif type(ir.dobject) == Pointer:
                return f'{ir.dobject.dtype} {ir.dobject.name()};\n'
            
            elif type(ir.dobject) == Buffer or Uniq:
                return f'torch::Tensor {ir.dobject.dobject.__name__}_{ir.dobject.__class__.__name__} = torch::empty({{{to_string(ir.dobject.dobject.size[0])}/16, 16}}, torch::TensorOptions(torch::k{"Int" if ir.dobject.dobject.dtype=="int" else "Float"}).device(torch::kCUDA));\n'
            else:
                return f'{to_string(ir.dobject)}'
            # elif type(ir.dobject) == Ref:
            #     code = f'{ir.dobject.dobject.dtype}* {ir.dobject.name()} = ({ir.dobject.dobject.dtype}*)&{ir.dobject.dobject.addr()}'
            #     return code
        case 'ThreadIdy' | 'ThreadIdx' | 'BlockIdy' | 'BlockIdx' | 'BlockDimy' | 'BlockDimx' | 'SyncThreads' | 'SyncWarps':
            return ir2gpu(ir)
        case 'ShuffleDown':
            return f'for (int off = blockDim.x/2; off > 0; off >>= 1) {{\n {to_string(ir.dobject)} += __shfl_down_sync(0xffffffff, {to_string(ir.dobject)}, off); \n}}\n'
        case 'ShuffleUp':
            return f'for (int off = blockDim.x/2; off > 0; off >>= 1) {{\n {to_string(ir.dobject)} += __shfl_up_sync(0xffffffff, {to_string(ir.dobject)}, off); \n}}\n'
        case 'ShuffleXor':
            return f'for (int off = blockDim.x/2; off > 0; off >>= 1) {{\n {to_string(ir.dobject)} += __shfl_xor_sync(0xffffffff, {to_string(ir.dobject)}, off); \n}}\n'
        case 'BroadCast':
            return f'{to_string(ir.dobject)} = __shfl_sync(0xffffffff, {to_string(ir.dobject)}, 0);\n'
        case 'SaveAtThread':
            return f'if (threadIdx.x == {ir.threadid}) {{\n {to_string(Assignment(ir.dst, ir.src))} }}\n'
        case 'Uniq' | 'Buffer':
            return f'{ir.dobject.__name__}_{ir.__class__.__name__}[{to_string(BlockIdx())}]'
            # return f'[{to_string(BlockIdx())}]' 
        case 'IF':
            return f"{to_string(ir.left)} = {to_string(ir.condition)} ? {to_string(ir.true_var)} : {to_string(ir.false_var)};\n"
        case 'Pointer':
            return f'{ir.name()}'
        case 'Access_ptr':
            code = f'{to_string(ir.dobject)}['
            for i in range(len(ir.idx)-1):
                code += '('
            for i in range(len(ir.idx)):
                if i<len(ir.idx)-1:
                    code += f'{to_string(ir.idx[i])}) * {to_string(ir.dobject.size[i+1])}'
                    code += '+'
                else:
                    code += f'{to_string(ir.idx[i])}'
            
            code += ']'
            return code
        case _:
            return str(ir)

def gen_cuda(ast, cpu_ir, gpu_ir):
    # 2 ir list for cpu and gpu
    def action_cpu(node, res):
        if node.valid:
            if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
                res.extend(node.decl)
            elif type(node) == TensorOp:
                res.extend(node.decl)
                # res.extend(node.compute)


    def action_cuda(node, res):
        if node.valid:
            if type(node) == TensorOp:
                # res.extend(node.decl)
                res.extend(node.compute)

    t = helpers.ASGTraversal(action_cpu)
    cpu_ir.extend(t(ast))

    t = helpers.ASGTraversal(action_cuda)
    gpu_ir.extend(t(ast))

def print_cuda(ast):
    cpu_ir = []
    gpu_ir = []
    gen_cuda(ast, cpu_ir, gpu_ir)
    # print(cpu_ir, "CUDA IR:::" , gpu_ir)

    args = helpers.get_input_nodes(ast)
    
    argscpu = ', '.join([f'torch::Tensor obj_{a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])
    # argsptr = ', '.join([f'obj_{a}.data_ptr<{args[a].dtype}>()' if type(args[a]) == Tensor else f'{a}' for a in args])
    # ptrs = ', '.join([f'{args[a].dtype}* {a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])
    
    argsptr = ', '.join([f'obj_{a}.packed_accessor32<{args[a].dtype if args[a].dtype!="int" else "int64_t"}, {len(args[a].ref_size)}, torch::RestrictPtrTraits>()' if type(args[a]) == Tensor else f'{a}' for a in args])
    ptrs = ', '.join([f'torch::PackedTensorAccessor32<{args[a].dtype if args[a].dtype!="int" else "int64_t"}, {len(args[a].ref_size)}, torch::RestrictPtrTraits> {a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])
    # in cuda kernel:     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
    # host call cuda:     .packed_accessor32<float, 2, torch::RestrictPtrTraits>()

    code = ''
    declare = ''
    for d in gpu_ir:
        if d:
            if type(d) == Decl and type(d.dobject) == Ndarray:
                declare += to_string(d)
                argsptr += f', obj_{d.dobject.name()}.packed_accessor32<{d.dobject.dtype}, {len(d.dobject.size)}, torch::RestrictPtrTraits>()'
                ptrs += f', torch::PackedTensorAccessor32<{d.dobject.dtype}, {len(d.dobject.size)}, torch::RestrictPtrTraits> {d.dobject.name()}'
            elif type(d) == Decl and type(d.dobject) in [Buffer, Uniq]:
                declare += to_string(d)
                argsptr += f', {d.dobject.dobject.__name__}_{d.dobject.__class__.__name__}.packed_accessor32<{d.dobject.dobject.dtype}, 2, torch::RestrictPtrTraits>()'
                ptrs += f', torch::PackedTensorAccessor32<{d.dobject.dobject.dtype}, 2, torch::RestrictPtrTraits> {d.dobject.dobject.__name__}_{d.dobject.__class__.__name__}'
            else:
                code += to_string(d)
    # print(declare)
    Return = ''
    if type(ast.eval) == Scalar:
        rtype = ast.dtype
        Return = f'return {ast.eval.name()};\n'
    elif type(ast.eval) == Ndarray:
        rtype = 'torch::Tensor'
        Return = f'return obj_{ast.eval.name()};\n'
    else:
        raise TypeError('wrong output type', ast.eval)

    with open('codegen/gpu_template.cu', 'r') as f:
        c_code = f.read()
        c_code = c_code.replace('RTYPE', rtype).replace('FNAME', ast.name).replace('ARGS', argscpu).replace('CODE', code).replace('PTR_VARS', argsptr).replace('PTRS', ptrs).replace('DECL', declare).replace('RETURN', Return)
    return c_code