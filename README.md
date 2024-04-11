# Cuke
Cuke is a source-to-source compiler that translates tensor computations written in Python into C++/CUDA code.
It was initially developed to teach compiler optimization at the University of Iowa and has since evolved into a platform for constructing domain-specific compilers  for different applications.


## Installation
Make sure to use python 3.8 or later:
```cmd
conda create -n cuke python=3.8
conda activate cuke 
```
Check out and install this repository: TODO: add a setup.py to the repo for installation.
```cmd
git clone https://github.com/pengjiang-hpc/cuke
pip install -e cuke
```

## Usage

TODO (Yihua): 1) add a simple example: A + B to show how to import the Tensor classes in asg.py...

2) add an example of set intersection using cond apply


More examples can be found in the ``apps`` folder. 


## Comparison with Other Tools
**Q: How is cuke different from Python libraries such as numpy/scipy/pytorch?**

A: The main difference is that cuke is a compiler. Instead of calling pre-compiled code, it generates source code (e.g., C++, CUDA) that runs on different hardware. The code generation is achieved through an intermediate representation, which allows users to apply various optimization transformations, such as loop fusion, parallelization, data buffering, etc. As a result, the generated code often achieves better performance than library-based solutions. Extending cuke to support new hardware is also much easier as it only requires implementation of a new backend. 

**Q: How is cuke different from ML compilers such as TVM/XLA/TorchScript?**

A: Cuke is more focused on supporting applications with irregular computation and memory access patterns. Its main differences with TVM/XLA/TorchScript include: 
1) It supports a more general syntax than basic tensor algebra and thus can express more general computations than neural networks. 
2) It allows users to compose customized operators using the basic syntax, enabling more aggressive code optimization based on application-specific information. For example, our IPDPS'24 paper shows that cuke can perform aggressive loop fusions that TVM/XLA/TorchScript ignores. 
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