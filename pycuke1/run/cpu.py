from torch.utils.cpp_extension import load
import os

def compile_and_run(code, *args):
    path = f'{os.path.dirname(__file__)}/.tmp'
    if not os.path.exists(path):
        os.mkdir(path)
    file = os.path.join(path, 'cpu_code.cpp')
    f = open(file, 'w')
    f.write(code)
    f.close()
    module = load(name='module', sources=[file])
    return module.run(*args)