int einsum_apply__edge_list_iinczwxr(int num_jobs, torch::Tensor obj_edge_list, int num_node, torch::Tensor obj_rowptr, int num_edge, torch::Tensor obj_colidx)
{
    auto edge_list = obj_edge_list.accessor<int, 2>();
int s10;
s10 = (num_node + 1);
auto rowptr = obj_rowptr.accessor<int, 1>();
int s38;
auto colidx = obj_colidx.accessor<int, 1>();
int s37;
int s43;
s43 = 0;
torch::Tensor obj_arr45 = torch::empty({4096}, at::kInt);
auto arr45 = obj_arr45.accessor<int, 1>();
int s226;
int s76;
int s75;
int s81;
s81 = 0;
torch::Tensor obj_arr83 = torch::empty({4096}, at::kInt);
auto arr83 = obj_arr83.accessor<int, 1>();
int s227;
int s103;
int s108;
s108 = 0;
torch::Tensor obj_arr110 = torch::empty({(s81 - 0)}, at::kInt);
auto arr110 = obj_arr110.accessor<int, 1>();
int s228;
int s141;
int s140;
int s146;
s146 = 0;
torch::Tensor obj_arr148 = torch::empty({4096}, at::kInt);
auto arr148 = obj_arr148.accessor<int, 1>();
int s229;
int s168;
int s173;
s173 = 0;
torch::Tensor obj_arr175 = torch::empty({(s146 - 0)}, at::kInt);
auto arr175 = obj_arr175.accessor<int, 1>();
int s230;
int s195;
int s200;
s200 = 0;
torch::Tensor obj_arr202 = torch::empty({(s173 - 0)}, at::kInt);
auto arr202 = obj_arr202.accessor<int, 1>();
int s231;
int s232;
int s225;
int s233;
s225 = 0;
for (int _l17 = 0; _l17 < num_jobs; _l17 += 1) {
    s38 = (rowptr[(edge_list[_l17][0] + 1)] - rowptr[edge_list[_l17][0]]);
    s37 = (rowptr[(edge_list[_l17][1] + 1)] - rowptr[edge_list[_l17][1]]);
    for (int _l2 = 0; _l2 < (rowptr[(edge_list[_l17][0] + 1)] - rowptr[edge_list[_l17][0]]); _l2 += 1) {
        s43 = 0;
        s226 = BinarySearch(colidx[((rowptr[edge_list[_l17][1]])+(0))], 0, s37, colidx[((rowptr[edge_list[_l17][0]])+(_l2))]);
        if (s226) {
            arr45[s43] = colidx[((rowptr[edge_list[_l17][0]])+(_l2))];
            s43 += 1;
        } 
    } 
    s76 = (rowptr[(edge_list[_l17][0] + 1)] - rowptr[edge_list[_l17][0]]);
    s75 = (rowptr[(edge_list[_l17][1] + 1)] - rowptr[edge_list[_l17][1]]);
    for (int _l5 = 0; _l5 < (rowptr[(edge_list[_l17][0] + 1)] - rowptr[edge_list[_l17][0]]); _l5 += 1) {
        s81 = 0;
        s227 = BinarySearch(colidx[((rowptr[edge_list[_l17][1]])+(0))], 0, s75, colidx[((rowptr[edge_list[_l17][0]])+(_l5))]);
        if (s227) {
            arr83[s81] = colidx[((rowptr[edge_list[_l17][0]])+(_l5))];
            s81 += 1;
        } 
    } 
    s141 = (rowptr[(edge_list[_l17][0] + 1)] - rowptr[edge_list[_l17][0]]);
    s140 = (rowptr[(edge_list[_l17][1] + 1)] - rowptr[edge_list[_l17][1]]);
    for (int _l10 = 0; _l10 < (rowptr[(edge_list[_l17][0] + 1)] - rowptr[edge_list[_l17][0]]); _l10 += 1) {
        s146 = 0;
        s229 = BinarySearch(colidx[((rowptr[edge_list[_l17][1]])+(0))], 0, s140, colidx[((rowptr[edge_list[_l17][0]])+(_l10))]);
        if (s229) {
            arr148[s146] = colidx[((rowptr[edge_list[_l17][0]])+(_l10))];
            s146 += 1;
        } 
    } 
    s233 = 0;
    for (int _l16 = 0; _l16 < (s43 - 0); _l16 += 1) {
        s103 = (rowptr[(arr45[(_l16)] + 1)] - rowptr[arr45[(_l16)]]);
        for (int _l7 = 0; _l7 < (s81 - 0); _l7 += 1) {
            s108 = 0;
            s228 = BinarySearch(colidx[((rowptr[arr45[(_l16)]])+(0))], 0, s103, arr83[(_l7)]);
            if (s228) {
                arr110[s108] = arr83[(_l7)];
                s108 += 1;
            } 
        } 
        s168 = (rowptr[(arr45[(_l16)] + 1)] - rowptr[arr45[(_l16)]]);
        for (int _l12 = 0; _l12 < (s146 - 0); _l12 += 1) {
            s173 = 0;
            s230 = BinarySearch(colidx[((rowptr[arr45[(_l16)]])+(0))], 0, s168, arr148[(_l12)]);
            if (s230) {
                arr175[s173] = arr148[(_l12)];
                s173 += 1;
            } 
        } 
        s232 = 0;
        for (int _l15 = 0; _l15 < (s108 - 0); _l15 += 1) {
            s195 = (rowptr[(arr110[(_l15)] + 1)] - rowptr[arr110[(_l15)]]);
            for (int _l14 = 0; _l14 < (s173 - 0); _l14 += 1) {
                s200 = 0;
                s231 = BinarySearch(colidx[((rowptr[arr110[(_l15)]])+(0))], 0, s195, arr175[(_l14)]);
                if (s231) {
                    arr202[s200] = arr175[(_l14)];
                    s200 += 1;
                } 
            } 
            s232 += (s200 - 0);
        } 
        s233 += s232;
    } 
    s225 += s233;
} 
return s225;
