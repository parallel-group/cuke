from torch.utils.cpp_extension import load
import os

def compile_and_run(code, *args):
    filename = 'run/.tmp/cuda_code.cu'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
            if content != code:
                f = open(filename, 'w')
                f.write(code)
                f.close()
    else:
        if not os.path.exists('run/.tmp'):
            os.makedirs('run/.tmp')
        f = open(filename, 'w')
        f.write(code)
        f.close()
    module = load(name='module', sources=[filename])
    return module.run(*args)