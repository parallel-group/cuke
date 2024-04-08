# Automatic Efficient Code Generator for Knowledge Graph Embedding Score Function

This is a compiler tool for automatic code generation of Knowledge Graph Embedding (KGE) score function computation based on *loop fusion and memory optimization*. 
Please see our IPDPS24 paper "cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedding." for the theories and algorithms. 
The tool basically has two parts: an Abstract Syntax Graph (ASG) and IR pass based on ASG that analyzes and derives user-input Python code, and a code generation tool that produces the cpu and gpu code based on the analysis results. 


## Appetizer
Take a look at a simple example of TransE:

```python
Eemb = Tensor((nnodes, dim), name='Eemb')
Remb = Tensor((nedges, dim), name='Remb')
h = Tensor((batch_size, ), dtype='int', name='h')
t = Tensor((batch_size, ), dtype='int', name='t')
r = Tensor((batch_size, ), dtype='int', name='r')
res = Eemb[h] - Eemb[t] + Remb[r]
code = codegen.gpu.print_cuda(gen_ir(res))
print(code)
```

This score function computation quantifies how well the relation vector translates the head entity vector to the tail entity vector. For the given arguments `Eemb`, `Remb`, `h`, `t`, `r` (all of them are tensors), cuKE would generate computational graph node in the ASG and transform it into our defined IR. The generated IR can be transformed based on our transformation and finally generate machine-executable code. The generated CUDA code in TransE looks something like:

```cpp
// split batch_size for block parallel
for (int _l4 = (blockIdx.x * blockDim.y); _l4 < ((blockIdx.x * blockDim.y) + (batch_size / (batch_size/16))); _l4 += 16) {
  // split batch_size for warp parallel
  for (int _l5 = (_l4 + threadIdx.y); _l5 < ((_l4 + 16) < batch_size ? ((_l4 + 16)) : (batch_size)); _l5 += blockDim.y) {
    for (int _l6 = 0; _l6 < dim; _l6 += 64) {
      // split dim for thread parallel
      for (int _l7 = (_l6 + threadIdx.x); _l7 < ((_l6 + 64) < dim ? ((_l6 + 64)):(dim)); _l7 += blockDim.x) {
        // fuse two element-wise computation
        arr38[_l5][_l7] = ((Eemb[h[_l5]][(_l7)] - Eemb[t[_l5]][(_l7)]) + Remb[r[_l5]][(_l7)]); 
      } 
    } 
  } 
}
```

For the sampled graph, if some relation embeddings can be reused, we can store them into CUDA shared memory for better memory access, we can simply apply this transformation by adding:
```python
r.attr['reuse'] = True
```

The code is then changed to:
```cpp
__shared__ float arr66[2][64];
float s89;
for (int _l4 = (blockIdx.x * blockDim.y); _l4 < ((blockIdx.x * blockDim.y) + (batch_size / (batch_size/16))); _l4 += 16) {
  for (int _l5 = (_l4 + threadIdx.y); _l5 < ((_l4 + 16) < batch_size ? ((_l4 + 16)) : (batch_size)); _l5 += blockDim.y) {
      for (int _l6 = 0; _l6 < dim; _l6 += 64) {
        for (int _l8 = (threadIdx.x + (threadIdx.y * blockDim.x)); _l8 < (r_unique_cnt[blockIdx.x] * 64); _l8 += (blockDim.x * blockDim.y)) {
          // stores reused relation embedding into shared memory
          arr66[(_l8 / 64)][(_l8 % 64)] = Remb[r_uniq[blockIdx.x][(_l8 / 64)]][((_l8 % 64) + _l6)];
        } 
        __syncthreads();
        for (int _l7 = (_l6 + threadIdx.x); _l7 < ((_l6 + 64) < dim ? ((_l6 + 64)) : (dim)); _l7 += blockDim.x) {
          // access reused relation embedding or access global relation embedding
          s89 = ((r_buf[blockIdx.x][threadIdx.y] < 16) ? arr66[r_buf[blockIdx.x][threadIdx.y]][(_l7 - _l6)] : Remb[(r_buf[blockIdx.x][threadIdx.y] - 16)][_l7]);
          arr38[_l5][_l7] = ((Eemb[h[_l5]][(_l7)] - Eemb[t[_l5]][(_l7)]) + s89);
      } 
    } 
  } 
}
```


The code generation of the TransE is quite simple. Now take a look at more complex example on TransR:

```python
Eemb = Tensor((nnodes, dim), name='Eemb')
Remb = Tensor((nedges, dim), name='Remb')
Proj = Tensor((nedges, dim, dim), name='Proj')
h = Tensor((batch_size, ), dtype='int', name='h')
t = Tensor((batch_size, ), dtype='int', name='t')
r = Tensor((batch_size, ), dtype='int', name='r')
r.attr['reuse'] = True
res = bvm(Eemb[h] - Eemb[t], Proj[r]) + Remb[r]
code = codegen.gpu.print_cuda(gen_ir(res))
print(code)
```

In TransR, the head and tail entity embeddings are first transformed by the respective relation-specific projection matrices `Proj`. Then, the transformed entity embeddings are combined with the relation embedding, and the distance between these two vectors is computed. The generated CUDA code is:

```cpp
__shared__ float arr158[2][64][64];
__shared__ float arr187[16][64];
__shared__ float arr207[2][64];

for (int _l7 = (blockIdx.x * blockDim.y); _l7 < ((blockIdx.x * blockDim.y) + (batch_size / (batch_size/16))); _l7 += 16) {
  for (int _l8 = (_l7 + threadIdx.y); _l8 < ((_l7 + 16) < batch_size ? ((_l7 + 16)) : (batch_size)); _l8 += blockDim.y) {
    for (int _l9 = 0; _l9 < dim; _l9 += 64) {
      for (int _l10 = (_l9 + threadIdx.x); _l10 < ((_l9 + 64) < dim ? ((_l9 + 64)) : (dim)); _l10 += blockDim.x) {
        // compute elementwise operation
        arr22[_l8][_l10] = (Eemb[h[_l8]][(_l10)] - Eemb[t[_l8]][(_l10)]);
      } 
    } 
  } 
} 
float s183;
float s230;
for (int _l11 = (blockIdx.x * blockDim.y); _l11 < ((blockIdx.x * blockDim.y) + (batch_size / (batch_size/16))); _l11 += 16) {
  for (int _l12 = (_l11 + threadIdx.y); _l12 < ((_l11 + 16) < batch_size ? ((_l11 + 16)) : (batch_size)); _l12 += blockDim.y) {
    for (int _l13 = 0; _l13 < dim; _l13 += 64) {
      for (int _l20 = (threadIdx.x + (threadIdx.y * blockDim.x)); _l20 < (r_unique_cnt[blockIdx.x] * 64); _l20 += (blockDim.x * blockDim.y)) {
        // store vector into shared memory
        arr207[(_l20 / 64)][(_l20 % 64)] = Remb[r_uniq[blockIdx.x][(_l20 / 64)]][((_l20 % 64) + _l13)];
      } 
      __syncthreads();
      for (int _l14 = (_l13 + threadIdx.x); _l14 < ((_l13 + 64) < dim ? ((_l13 + 64)) : (dim)); _l14 += blockDim.x) {
        arr187[_l12-_l11][_l14-_l13] = 0;
      } 
      for (int _l15 = 0; _l15 < (dim - 0); _l15 += 64) {
        for (int _l14 = (_l13 + threadIdx.x); _l14 < ((_l13 + 64) < dim ? ((_l13 + 64)) : (dim)); _l14 += blockDim.x) {
          for (int _l17 = 0; _l17 < 2; _l17 += 1) {
            for (int _l18 = threadIdx.y; _l18 < 64; _l18 += blockDim.y) {
              for (int _l19 = threadIdx.x; _l19 < 64; _l19 += blockDim.x) {
                // store matrix into shared memory
                arr158[_l17][_l18][_l19] = Proj[r_uniq[blockIdx.x][_l17]][(_l18 + _l13)][(_l19 + _l15)];
              } 
            } 
          } 
          __syncthreads();
          for (int _l16 = _l15; _l16 < ((_l15 + 64) < (dim - 0) ? ((_l15 + 64)) : ((dim - 0))); _l16 += 1) {
            // batched vector-matrix multiplication
            arr187[_l12-_l11][_l14-_l13] += (arr22[_l12][(_l16)] * s183);
          } 
        } 
      } 
      for (int _l14 = (_l13 + threadIdx.x); _l14 < ((_l13 + 64) < dim ? ((_l13 + 64)) : (dim)); _l14 += blockDim.x) {
        s230 = ((r_buf[blockIdx.x][threadIdx.y] < 16) ? arr207[r_buf[blockIdx.x][threadIdx.y]][(_l14 - _l13)] : Remb[(r_buf[blockIdx.x][threadIdx.y] - 16)][_l14]);
        arr75[_l12][_l14] = (arr187[_l12-_l11][_l14-_l13] + s230);
      } 
    } 
  } 
}
```

You can see that our fusion strategies are suitable for KGE score function and the generated code is effectively parallelized and the CUDA characteristic is well-used. For more information, please read our paper:

*cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedding. Lihan Hu, Jing Li, Peng Jiang. IPDPS 2024.* 

And you will find the steps and algorithms to parallelize such loops are actually very straightforward. 
Our tool generates the above parallel codes automatically, and it can parallelize more complicated loops than this simple example.  



## Install
cuKE is easy to install, we provide the requirement lists, so you just need run as follows:

```cmd
git clone https://github.com/pengjiang-hpc/cuke
cd cuke
conda create --name YOUR_ENV_NAME --file requirements.txt python=3.8
conda activate YOUR_ENV_NAME
```

Congratulations! You are ready to use our code generation tool cuKE!


## Note
The tool is currently a prototype, especially the code genenration and transformation part. 
We have tested it with all the score functions provided in `kge.py`. 
More engineering efforts and testing are needed to optimize and stablize it.  
Contact the author (lihan-hu@uiowa.edu) or open an issue in this repository if you have any questions or suggestions.

## Reference
```bibtex
@article{hu2023cuke,
  author    = {Lihan Hu and Jing Li and Peng Jiang},
  title     = {cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedding},
  booktitle = {38th IEEE International Parallel & Distributed Processing Symposium (IPDPS)},
  year = {2024}
}
```