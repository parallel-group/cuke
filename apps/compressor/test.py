from pycuke.asg import *
import pycuke.codegen as codegen
from pycuke.asg2ir import gen_ir
import torch
import pycuke.run as run

def truncate(data, nbits):
    src = '''unsigned m = 0x0000FFFF;
        for (int j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
            for (int k = 0; k < 32; k = (k + j + 1) & ~j) {
                unsigned t = (DATA[k] ^ (DATA[k+j] >> j)) & m;
                DATA[k] = DATA[k] ^ t;
                DATA[k+j] = DATA[k+j] ^ (t << j);
            }
        }'''
    return inline(src, [('DATA', data)], [('DATA', data)])

def test_compressor():
    block_size = Var(name='block_size')
    nblocks = setval(1024//block_size, name='nblocks')
    input = Tensor((1024, ), name='input', dtype='float')
    input = input.view((nblocks, block_size), [0, 1])
    quant = (input * 1000).round()
    lorenzo_res = quant[:, 0:32] - quant[:, -1:31]
    encode_nbits = lorenzo_res.abs().max(axis=1).nbits()
    ofs = encode_nbits.prefix_sum(inclusive=False)
    compressed_res = apply(lambda x, y: truncate(x, y), (lorenzo_res, encode_nbits), out_ofs=ofs)

    res = compressed_res

    code = codegen.cpu.print_cpp(gen_ir(res))

    print(code)


if __name__ == '__main__':
    test_compressor()