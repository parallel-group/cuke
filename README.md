# Cuke
Cuke is a source-to-source compiler that translates tensor computations written in Python into C++/CUDA code.
It was initially developed to teach compiler optimization at the University of Iowa but has since evolved into a platform for constructing domain-specific compilers  for different applications.


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
- How is cuke different from Python libraries such as numpy/scipy/pytorch?

- How is cuke different from ML compilers such as TVM/XLA? 
  

## Citation
```bibtex
@article{hu2023cuke,
  author    = {Lihan Hu and Jing Li and Peng Jiang},
  title     = {cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedding},
  booktitle = {38th IEEE International Parallel & Distributed Processing Symposium (IPDPS)},
  year = {2024}
}
```