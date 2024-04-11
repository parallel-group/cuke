#include <torch/extension.h>

__global__ void FNAME_kernel(PTRS){
    CU_DE
    CODE
}

RTYPE FNAME(ARGS)
{   
    DECL
    FNAME_kernel<<< block, dim3(tx,ty) >>>(PTR_VARS);
    RETURN
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &FNAME);
}