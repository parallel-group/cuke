#include <torch/extension.h>

bool BinarySearch(int& arr, int start, int end, int target){
    int* nums = &arr;

    size_t mid;
    size_t low = start;
    size_t high = end;

    while (low < high) {
        mid = low + (high - low) / 2;
        if (target <= nums[mid]) {
            high = mid;

        }
        else {
            low = mid + 1;
        }
    }
    if(low < end && nums[low] < target) {
       low++;
    }
    if(low>=start && low < end &&  nums[low]==target) return true;
    else return false;
}

RTYPE FNAME(ARGS)
{
    CODE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &FNAME);
}