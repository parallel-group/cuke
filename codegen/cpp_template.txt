#include <torch/extension.h>

RTYPE FNAME(ARGS)
{
    CODE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &FNAME);
}