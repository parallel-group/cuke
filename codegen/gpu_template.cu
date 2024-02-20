#include <torch/extension.h>

int Hashing(int vid, auto& hash_table){
    return vid*2;
}

inline size_t SetIntersection(int& first_arr, size_t first_size, int& second_arr, size_t second_size, int& res_arr)
{
  int* first = &first_arr;
  int* second = &second_arr;
  int* res = &res_arr;
  size_t pi = 0, pj = 0, pos = 0;
  while (pi != first_size && pj != second_size) {
    if (first[pi] < second[pj])
      pi++;
    else if (first[pi] > second[pj])
      pj++;
    else {
      res[pos++] = first[pi];
      pi++;
      pj++;
    }
  }
  return pos;
}


inline size_t SetDifference(int& first_arr, size_t first_size, int& second_arr, size_t second_size, int& res_arr){
  int* first = &first_arr;
  int* second = &second_arr;
  int* res = &res_arr;
  size_t pi = 0, pj = 0, pos = 0;
  while (pi != first_size && pj!=second_size){
      int left = first[pi]; 
      int right = second[pj];
      if(left<=right) pi++;
      if(right<=left) pj++;
      
      if (left < right) {
        res[pos++] = left;
      }
  }
  while(pi<first_size){
    int left = first[pi++];
        res[pos++]=left;
  }
  return pos;
}


__global__ void FNAME_kernel(PTRS){
    CU_DE
    CODE
}

RTYPE FNAME(ARGS)
{   
    DECL
    FNAME_kernel<<< block, dim3(ty,tx) >>>(PTR_VARS);
    RETURN
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &FNAME);
}