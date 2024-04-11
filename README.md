# Cuke
Cuke is a source-to-source compiler that translates tensor computations written in Python into C++/CUDA code.
It was initially developed to teach compiler optimization at the University of Iowa and has since evolved into a platform for constructing domain-specific compilers  for different applications.


## Installation
Make sure to use python 3.8 or later:
```cmd
conda create -n cuke python=3.8
conda activate cuke 
```
Check out and install this repository:
```cmd
git clone https://github.com/pengjiang-hpc/cuke
pip install cuke/
```

You can also use ``python setup.py install`` to install cuke


## Usage
**An example of elementwise add**
```python
import cuke.codegen as codegen
from cuke.asg import *
from cuke.asg2ir import gen_ir

#Create two tensor nodes: A and B
A = Tensor((10, ))
B = Tensor((10, ))

#Create an elementwise add operator.
#A and B are the input nodes, res is the output node. 
res = A + B

#Now we get an ASG of three tensor nodes.
#`gen_ir` invokes the asg->ir procedure and `print_cpp` returns the generated C++ code. 
code = codegen.cpu.print_cpp(gen_ir(res))
print(code)
```
**An example of set intersection using cond apply**
 ```python
def is_in(x, li):
    src = inspect.cleandoc("""
    F = BinarySearch(LI, 0, LSIZE, X);
    """)
    found = Var(dtype='int')
    found.attr['is_arg'] = False
    return inline(src, [('F', found)], [('X', x), ('LI', li), ('LSIZE', li._size()[0])])

def intersect(a, b):
    #We create an apply operator, a is the input and cond is the output.
    #The 'apply' operator invokes the 'is_in' function for each element of 'a'.
    #The cond has the same size as a, and stores the result of the is_in function for each element of a in the corresponding position.
    cond = a.apply(lambda x: is_in(x, b))
    #We creats an conditional apply opearator.
    #For each element 'x' of a(x=a[i], i is the iterator), if cond[i] is true, we make an assignment c[csize++]=a[i].
    #The size of c(csize) is not the same as a 
    c = a.apply(lambda x: x, cond=cond)
    return c

A = Tensor((10, ))
B = Tensor((20, ))
res = intersect(A, B)
code = codegen.cpu.print_cpp(gen_ir(res))
print(code)
 ```

More examples can be found in the ``apps`` folder. 


## Comparison with Other Tools
**Q: How is cuke different from Python libraries such as numpy/scipy/pytorch?**

A: The main difference is that cuke is a compiler. Instead of calling pre-compiled code, it generates source code (e.g., C++, CUDA) that runs on different hardware. The code generation is achieved through an intermediate representation, which allows users to apply various optimization transformations, such as loop fusion, parallelization, data buffering, etc. As a result, the generated code often achieves better performance than library-based solutions. Extending cuke to support new hardware is also much easier as it only requires implementation of a new backend. 

**Q: How is cuke different from ML compilers such as TVM/XLA?**

A: Cuke is more focused on supporting applications with irregular computation and memory access patterns. Its main differences with TVM/XLA include: 
1) It supports a more general syntax than basic tensor algebra and thus can express more general computations than neural networks. 
2) It allows users to compose customized operators using the basic syntax, enabling more aggressive code optimization based on application-specific information. For example, our IPDPS'24 paper shows that cuke can perform aggressive loop fusions that TVM/XLA ignores. 
3) It supports inspector-executor compilation for indirect tensor indexing to reduce memory access overhead. 

  

## Citation
```bibtex
@article{hu2023cuke,
  author    = {Lihan Hu and Jing Li and Peng Jiang},
  title     = {cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedding},
  booktitle = {38th IEEE International Parallel & Distributed Processing Symposium (IPDPS)},
  year = {2024}
}
```
