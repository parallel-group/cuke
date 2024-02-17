int einsum_apply__edge_list_zjlayauj(int num_jobs, torch::Tensor obj_edge_list, int num_node, torch::Tensor obj_rowptr, int num_edge, torch::Tensor obj_colidx)
{
    auto edge_list = obj_edge_list.accessor<int, 2>();
int s9;
auto rowptr = obj_rowptr.accessor<int, 1>();
auto colidx = obj_colidx.accessor<int, 1>();
torch::Tensor obj_arr7 = torch::empty({4096}, at::kInt);
auto arr7 = obj_arr7.accessor<int, 1>();
int s187;
int s49;
int s53;
torch::Tensor obj_arr51 = torch::empty({4096}, at::kInt);
auto arr51 = obj_arr51.accessor<int, 1>();
torch::Tensor obj_arr47 = torch::empty({4096}, at::kInt);
auto arr47 = obj_arr47.accessor<int, 1>();
int s180;
int s104;
int s108;
int s112;
torch::Tensor obj_arr110 = torch::empty({4096}, at::kInt);
auto arr110 = obj_arr110.accessor<int, 1>();
torch::Tensor obj_arr106 = torch::empty({4096}, at::kInt);
auto arr106 = obj_arr106.accessor<int, 1>();
torch::Tensor obj_arr102 = torch::empty({4096}, at::kInt);
auto arr102 = obj_arr102.accessor<int, 1>();
int s198;
int s197;
int s199;
s197 = 0;
for (int _l5 = 0; _l5 < num_jobs; _l5 += 1) {
s9 = SetIntersection(colidx[((rowptr[edge_list[_l5][0]])+(0))], (rowptr[(edge_list[_l5][0] + 1)] - rowptr[edge_list[_l5][0]]), colidx[((rowptr[edge_list[_l5][1]])+(0))], (rowptr[(edge_list[_l5][1] + 1)] - rowptr[edge_list[_l5][1]]), arr7[0]);
s187 = (s9 - 0);
s199 = 0;
for (int _l4 = 0; _l4 < (s9 - 0); _l4 += 1) {
s53 = SetIntersection(colidx[((rowptr[edge_list[_l5][0]])+(0))], (rowptr[(edge_list[_l5][0] + 1)] - rowptr[edge_list[_l5][0]]), colidx[((rowptr[edge_list[_l5][1]])+(0))], (rowptr[(edge_list[_l5][1] + 1)] - rowptr[edge_list[_l5][1]]), arr51[0]);
s49 = SetIntersection(arr51[(0)], (s53 - 0), colidx[((rowptr[arr7[(_l4)]])+(0))], (rowptr[(arr7[(_l4)] + 1)] - rowptr[arr7[(_l4)]]), arr47[0]);
s180 = (s49 - 0);
s198 = 0;
for (int _l3 = 0; _l3 < (s49 - 0); _l3 += 1) {
s112 = SetIntersection(colidx[((rowptr[edge_list[_l5][0]])+(0))], (rowptr[(edge_list[_l5][0] + 1)] - rowptr[edge_list[_l5][0]]), colidx[((rowptr[edge_list[_l5][1]])+(0))], (rowptr[(edge_list[_l5][1] + 1)] - rowptr[edge_list[_l5][1]]), arr110[0]);
s108 = SetIntersection(arr110[(0)], (s112 - 0), colidx[((rowptr[arr7[(_l4)]])+(0))], (rowptr[(arr7[(_l4)] + 1)] - rowptr[arr7[(_l4)]]), arr106[0]);
s104 = SetIntersection(arr106[(0)], (s108 - 0), colidx[((rowptr[arr47[(_l3)]])+(0))], (rowptr[(arr47[(_l3)] + 1)] - rowptr[arr47[(_l3)]]), arr102[0]);
s198 += (s104 - 0);
} 
s199 += s198;
} 
s197 += s199;
} 
return s197;

}