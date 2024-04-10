# An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedding

The artifact includes the code for reproducing the evaluation results in paper "cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedding".

## An Example
### Define a Score Function
Take a look at a simple example of the TransE score function:

```python
Eemb = Tensor((nnodes, dim), name='Eemb')
Remb = Tensor((nrel, dim), name='Remb')
h = Tensor((batch_size, ), dtype='int', name='h')
t = Tensor((batch_size, ), dtype='int', name='t')
r = Tensor((batch_size, ), dtype='int', name='r')
res = Eemb[h] - Eemb[t] + Remb[r]
```

Here, *Eemb* is the entity embedding vectors for all nodes in the graph, *Remb* is the relation embedding vectors for all relations in the graph, 
*nnodes* is the number of entities, and *nrel* is the number of relation types. The score function computes how well the relation vector translates the head entity vector to the tail entity vector.
### Generate the Intermediate Representation
As the computation is defined, cuKE generates a computational graph to represent the computation. 
We can then call the *gen_ir* function to generate a loop-based intermediate representation for the computation. 
```python
ir = gen_ir(res)
```
### Generate CPU Code
The generated IR can be translated into CPU or GPU code by calling the corresponding *codegen* functions.

```python
code = codegen.cpu.print_cpp(ir)
print(code)
```

The generated CPU code for the above TransE function looks something like:


<span style="color:red">TODO: add the original CPU code here. </span>

### Loop Fusion
The code has two loops. The first one computes *Eemb[h] - Eemb[t]*, and the second one adds the result of the first operator with *Remb[r]*.
The two loops can be easily fused into one loop by adding the *fuser* into the transform passes. The fused code looks like:

<span style="color:red">TODO: add the fused CPU code. </span>

While this fusion looks easy, our compiler supports more sophisticated loop fusions for many other score functions that traditional ML compiler such as TVM and XLA ignore. 
Please refer to our paper for details. 

### Parallelization for GPU

To generate efficient GPU code for the score function, we need to define a few parallelization/optimization passes to map the computation onto the thread hierarchy on a GPU and utilize the memory hierarchy to improve data access efficiency. 


<span style="color:red">TODO: explain the GPU code generation step by step. 1) loop tiling, 2) parallelization, 3) shared memory optimization</span>

The final code for the above TransE function looks something like:
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

### Runtime Inspection to Avoid Redundant Data Access
Since the number of relation types in a knowledge graph is much smaller than the number of edges, it is very likely that multiple edges in a sampled batch share the same relation types. 
In this case, the program only needs to load the embeddings of the unique relations from GPU global memory and stores the data in GPU shared memory for reuse. 
This optimization can be invoked by simply add a *reuse* attribute to the sampled relations:
```python
r.attr['reuse'] = True
```

The optimized code looks like:
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

There are implementations of more complicated KGE score functions in ``kge.py``. 



## Reproduce Results in Paper

<span style="color:red">TODO: see the example from Yihua's artifact: https://github.com/HPC-Research-Lab/STMatch </span>

