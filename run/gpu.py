from torch.utils.cpp_extension import load

def compile_and_run(code, *args):
    f = open('run/.tmp/cuda_code.cu', 'w')
    f.write(code)
    f.close()
    module = load(name='module', sources=['run/.tmp/cuda_code.cu'])
    return module.run(*args)