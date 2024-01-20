int einsum_apply__edge_list_fvygrvpp(int num_jobs, torch::Tensor obj_edge_list, int num_node, torch::Tensor obj_rowptr, int num_edge, torch::Tensor obj_colidx)
{
    auto edge_list = obj_edge_list.accessor<int, 2>();
    int s10;
    s10 = (num_node + 1);
    auto rowptr = obj_rowptr.accessor<int, 1>();
    int s38;
    auto colidx = obj_colidx.accessor<int, 1>();
    int s37;
    int s36;
    int s43;
    s43 = 0;
    torch::Tensor obj_arr45 = torch::empty({4096}, at::kInt);
    auto arr45 = obj_arr45.accessor<int, 1>();
    int s129;
    int s76;
    int s75;
    int s74;
    int s81;
    s81 = 0;
    torch::Tensor obj_arr83 = torch::empty({4096}, at::kInt);
    auto arr83 = obj_arr83.accessor<int, 1>();
    int s130;
    int s103;
    int s102;
    int s108;
    s108 = 0;
    torch::Tensor obj_arr110 = torch::empty({4096}, at::kInt);
    auto arr110 = obj_arr110.accessor<int, 1>();
    int s131;
    int s128;
    int s132;
    s128 = 0;
    for (int _l9 = 0; _l9 < num_jobs; _l9 += 1) {
        s38 = (rowptr[(edge_list[_l9][0] + 1)] - rowptr[edge_list[_l9][0]]);
        s37 = (rowptr[(edge_list[_l9][1] + 1)] - rowptr[edge_list[_l9][1]]);
        s43 = 0;
        for (int _l2 = 0; _l2 < (rowptr[(edge_list[_l9][0] + 1)] - rowptr[edge_list[_l9][0]]); _l2 += 1) {
            if(colidx[((rowptr[edge_list[_l9][0]])+(_l2))] < colidx[((rowptr[edge_list[_l9][1]])+(s36))]) continue; 
            else if (colidx[((rowptr[edge_list[_l9][0]])+(_l2))] > colidx[((rowptr[edge_list[_l9][1]])+(s36))]) { 
                while(s36 < s37 && colidx[((rowptr[edge_list[_l9][0]])+(_l2))] > colidx[((rowptr[edge_list[_l9][1]])+(s36))]){    
                    s36++;                                       
                } 
            } 
            if(s36 == s37) break;  
            if(colidx[((rowptr[edge_list[_l9][0]])+(_l2))]==colidx[((rowptr[edge_list[_l9][1]])+(s36))]){
                s129 = true;
            }
            else{
                s129 = false;
            }
            if (s129) {
            arr45[s43] = colidx[((rowptr[edge_list[_l9][0]])+(_l2))];
            s43 += 1;
            } 
        } 
        s76 = (rowptr[(edge_list[_l9][0] + 1)] - rowptr[edge_list[_l9][0]]);
        s75 = (rowptr[(edge_list[_l9][1] + 1)] - rowptr[edge_list[_l9][1]]);
        s81 = 0;
        for (int _l5 = 0; _l5 < (rowptr[(edge_list[_l9][0] + 1)] - rowptr[edge_list[_l9][0]]); _l5 += 1) {
        if(colidx[((rowptr[edge_list[_l9][0]])+(_l5))] < colidx[((rowptr[edge_list[_l9][1]])+(s74))]) continue; 
        else if (colidx[((rowptr[edge_list[_l9][0]])+(_l5))] > colidx[((rowptr[edge_list[_l9][1]])+(s74))]) { 
            while(s74 < s75 && colidx[((rowptr[edge_list[_l9][0]])+(_l5))] > colidx[((rowptr[edge_list[_l9][1]])+(s74))]){    
                s74++;                                       
            } 
        } 
        if(s74 == s75) break;  
        if(colidx[((rowptr[edge_list[_l9][0]])+(_l5))]==colidx[((rowptr[edge_list[_l9][1]])+(s74))]){
            s130 = true;
        }
        else{
            s130 = false;
        }
        if (s130) {
        arr83[s81] = colidx[((rowptr[edge_list[_l9][0]])+(_l5))];
        s81 += 1;
        } 
        } 
        s132 = 0;
        for (int _l8 = 0; _l8 < (s43 - 0); _l8 += 1) {
        s103 = (rowptr[(arr45[(_l8)] + 1)] - rowptr[arr45[(_l8)]]);
        s108 = 0;
        for (int _l7 = 0; _l7 < (s81 - 0); _l7 += 1) {
        if(arr83[(_l7)] < colidx[((rowptr[arr45[(_l8)]])+(s102))]) continue; 
        else if (arr83[(_l7)] > colidx[((rowptr[arr45[(_l8)]])+(s102))]) { 
            while(s102 < s103 && arr83[(_l7)] > colidx[((rowptr[arr45[(_l8)]])+(s102))]){    
                s102++;                                       
            } 
        } 
        if(s102 == s103) break;  
        if(arr83[(_l7)]==colidx[((rowptr[arr45[(_l8)]])+(s102))]){
            s131 = true;
        }
        else{
            s131 = false;
        }
        if (s131) {
        arr110[s108] = arr83[(_l7)];
        s108 += 1;
        } 
        } 
        s132 += (s108 - 0);
        } 
        s128 += s132;
    } 
    return s128;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &einsum_apply__edge_list_fvygrvpp);
}