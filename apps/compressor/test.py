from pycuke.asg import *
import pycuke.codegen as codegen
from pycuke.asg2ir import gen_ir
import torch
import pycuke.run as run


def test_compressor():
    block_size = Var(name='block_size')
    nblocks = setval(1024//block_size, name='nblocks')
    input = Tensor((1024, ), name='input', dtype='float')
    input = input.view((nblocks, block_size), [0, 1])
    quant = (input * 1000).round()

    lorenzo = quant[:, 0:32] - quant[:, -1:31]

    res = lorenzo

    code = codegen.cpu.print_cpp(gen_ir(res))

    print(code)


if __name__ == '__main__':
    test_compressor()