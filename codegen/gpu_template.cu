#include <torch/extension.h>

__global__ void FNAME_kernel(PTRS){
    CODE
}

RTYPE FNAME(ARGS)
{   
    DECL
    FNAME_kernel<<< batch_size/16, dim3(32,16) >>>(PTR_VARS);
    RETURN
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &FNAME);
}