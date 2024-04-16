from pycuke.asg import *
import pycuke.codegen as codegen
from pycuke.asg2ir import gen_ir
import torch
import pycuke.run as run


def sz():
    block_size = Var(name='block_size')
    nblocks = setval(1024//block_size, name='nblocks')
    input = Tensor((1024, ), name='input', dtype='float')
    input = input.view((nblocks, block_size), [0, 0])
    quant = (input * 1000).round()
    lorenzo_res = quant[:, 0:32] - quant[:, -1:31]
    encode_nbits = lorenzo_res.abs().max(axis=1).nbits()
    ofs = encode_nbits.prefix_sum(inclusive=False)

    res = inline('encoding(DATA, OFS);', [('DATA', lorenzo_res)], [('DATA', lorenzo_res), ('OFS', ofs)])
    res = mklist(res, ofs)

    code = codegen.cpu.print_cpp(gen_ir(res))

    print(code)


def regression():
    block_size = Var(name='block_size')
    nblocks = setval(1024 // block_size + 1, name='nblocks')
    input = Tensor((1024,), name='input', dtype='float')
    input = input.view((nblocks, block_size), [0, 0])
    quant_res = (input * 1000).round()

    res = quant_res.apply(lambda x: einsum('i,i->', x, Const(slice(1, block_size, 1), 'slice')))

    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)


def regression2():
    dim_size = Var(name='dim_size')
    block_size = dim_size * dim_size * dim_size
    nblocks = setval(1024 // block_size + 1, name='nblocks')
    input = Tensor((1024,), name='input', dtype='float')
    input = input.view((nblocks, dim_size, dim_size, dim_size), [0, 0, 0, 0])
    quant_res = (input * 1000).round()


    tmp1 = quant_res.apply(lambda x: einsum('ijk,i->', x, Const(slice(1, block_size, 1), 'slice')))
    tmp2 = quant_res.apply(lambda x: einsum('ijk,j->', x, Const(slice(1, block_size, 1), 'slice')))
    tmp3 = quant_res.apply(lambda x: einsum('ijk,k->', x, Const(slice(1, block_size, 1), 'slice')))
    tmp4 = quant_res.apply(lambda x: einsum('ijk,->', x, None))

    tmp = tmp1 + tmp2
    coefficient1 = tmp + tmp3
    coefficient2 = tmp + tmp4

    res = mklist(coefficient1, coefficient2)


    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)



if __name__ == '__main__':
    # sz()
    # regression()
    regression2()