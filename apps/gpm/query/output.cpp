
torch::Tensor apply__apply__index_Eembimveephw(int batch_size, int dim, int nnodes, torch::Tensor obj_Eemb, torch::Tensor obj_h, int nedges, torch::Tensor obj_Proj, torch::Tensor obj_r, torch::Tensor obj_t)
{
    auto Eemb = obj_Eemb.accessor<float, 2>();
    auto h = obj_h.accessor<int, 1>();
    auto Proj = obj_Proj.accessor<float, 3>();
    auto r = obj_r.accessor<int, 1>();
    auto t = obj_t.accessor<int, 1>();
    torch::Tensor obj_arr38 = torch::empty({batch_size}, at::kFloat);
    auto arr38 = obj_arr38.accessor<float, 1>();
    float s40;
    for (int _l3 = 0; _l3 < batch_size; _l3 += 1) {
        arr38[_l3] = 0;
        for (int _l4 = 0; _l4 < dim; _l4 += 1) {
            s40 = 0;
            for (int _l2 = 0; _l2 < dim; _l2 += 1) {
                s40 += (Eemb[h[_l3]][_l2] * Proj[r[_l3]][_l2][_l4]);
            } 
            arr38[_l3] += (s40 * Eemb[t[_l3]][_l4]);
        } 
    } 
    return obj_arr38;

}


torch::Tensor apply__apply__index_Eembjbxbfsjk(int batch_size, int dim, int nnodes, torch::Tensor obj_Eemb, torch::Tensor obj_h, int nedges, torch::Tensor obj_Proj, torch::Tensor obj_r, torch::Tensor obj_t)
{
    auto Eemb = obj_Eemb.accessor<float, 2>();
    auto h = obj_h.accessor<int, 1>();
    auto Proj = obj_Proj.accessor<float, 3>();
    auto r = obj_r.accessor<int, 1>();
    auto t = obj_t.accessor<int, 1>();
    torch::Tensor obj_arr69 = torch::empty({32,batch_size}, at::kFloat);
    auto arr69 = obj_arr69.accessor<float, 2>();
    torch::Tensor obj_arr68 = torch::empty({80,16,32}, at::kFloat);
    auto arr68 = obj_arr68.accessor<float, 3>();
    torch::Tensor obj_arr38 = torch::empty({batch_size}, at::kFloat);
    auto arr38 = obj_arr38.accessor<float, 1>();
    #pragma omp parallel for num_threads(80)
    for (int _l5 = 0; _l5 < batch_size; _l5 += 16) {
        #pragma omp parallel for num_threads(16)
        for (int _l6 = _l5; _l6 < ((_l5 + 16) < batch_size ? ((_l5 + 16)) : (batch_size)); _l6 += 1) {
            arr69[tid2][_l6] = 0;
            
            for (int _l7 = 0; _l7 < dim; _l7 += 64) {
                for (int _l8 = _l7; _l8 < ((_l7 + 64) < dim ? ((_l7 + 64)) : (dim)); _l8 += 1) {
                    
                    arr68[tid0][tid1][tid2] = 0;
                    for (int _l2 = 0; _l2 < dim; _l2 += 1) {
                        arr68[tid0][tid1][tid2] += (Eemb[h[_l6]][_l2] * Proj[r[_l6]][_l2][_l8]);
                    } 
                    
                    arr69[tid2][_l6] += (arr68[tid0][tid1][tid2] * Eemb[t[_l6]][_l8]);
                } 
            } 
            for (int64_t _l9 = 0; _l9 < 32; _l9 += 1) {
                arr38[_l6] += arr69[_l9][_l6];
            } 
        } 
    } 
    return obj_arr38;

}