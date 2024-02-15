import torch
import numpy as np
from itertools import permutations 

def read_graph(partial_edge):
    np_rowptr = np.fromfile("/workspace/Graphs/citeseer/snap.txt.vertex.bin", dtype=np.int64)
    np_colidx = np.fromfile("/workspace/Graphs/citeseer/snap.txt.edge.bin", dtype=np.int32)

    torch_rowptr = torch.from_numpy(np_rowptr, ).to(torch.int32)
    torch_colidx =torch.from_numpy(np_colidx)
    torch_edge_list = torch.zeros([torch_colidx.shape[0], 2], dtype=torch.int32)

    edge_idx = 0
    for i in range(0, torch_rowptr.shape[0]-1):
        for j in range(torch_rowptr[i].item() , torch_rowptr[i+1].item()):
                if (not partial_edge) or (partial_edge and torch_colidx[j]<i):
                    torch_edge_list[edge_idx][0] = i
                    torch_edge_list[edge_idx][1] = torch_colidx[j]
                    edge_idx = edge_idx+1
    #Cuke
    num_node = torch_rowptr.shape[0]-1
    num_edges = torch_colidx.shape[0]
    if partial_edge:
        num_jobs = edge_idx
    else:
        num_jobs = num_edges

    return torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs


def read_query_file(filename):
    pmtx = None
    num_node=0
    with open(filename) as p:
        for line in p:
            if line[0]=='v':
                num_node+=1
        p.seek(0, 0)

        pmtx = [[0 for y in range(num_node)] for x in range(num_node)]
        for line in p: 
            if line[0]=='e':
                v0 = int(line[2])
                v1 = int(line[4])
                pmtx[v0][v1] = 1
                pmtx[v1][v0] = 1
        return pmtx

def query_info(pmtx):
    num_edge = 0
    num_node = len(pmtx)
    edge_list = []
    for row_idx in range(0, len(pmtx)):
        for col_idx in range(0, len(pmtx[row_idx])):
            item = pmtx[row_idx][col_idx]
            if item==1 and row_idx<col_idx:
                edge_list.append([row_idx, col_idx])
                num_edge+=1
    
    return num_node, num_edge, edge_list
    

def query_dfs(pmtx):
    edge_order = []

    def _query_dfs(query_adjlist, cur_node, visited):
        if cur_node not in visited:
            visited.add(cur_node)
            for adj_node in query_adjlist[cur_node]:
                if [cur_node, adj_node] not in edge_order and [adj_node, cur_node] not in edge_order:
                    edge_order.append([cur_node, adj_node])
                _query_dfs(query_adjlist, adj_node, visited)
    
    query_adjlist = []

    for row_idx in range(0, len(pmtx)):
        this_list = []
        for col_idx in range(0, len(pmtx[row_idx])):
            if pmtx[row_idx][col_idx]==1:
                this_list.append(col_idx)
        
        query_adjlist.append(this_list)

    _query_dfs(query_adjlist, 0, set())
    return edge_order


def symmetry_breaking(pmtx):
    num_node = len(pmtx)
    perms = list(permutations(list(range(num_node))))

    valid_perms = []
    for perm in perms:
        mapped_adj = [None for x in range(num_node)]
        for i in range(num_node):
            tp = set()
            for j in range(num_node):
                if pmtx[i][j]==0:
                    continue
                tp.add(perm[j])
            mapped_adj[perm[i]] = tp

        valid = True
        for i in range(num_node):
            equal = True
            count = 0
            for j in range(num_node):
                if pmtx[i][j]==1:
                    count+=1
                    if j not in mapped_adj[i]:
                        equal=False
            
            if not equal or count!=len(mapped_adj[i]):
                valid = False
                break

        if valid==True:
            valid_perms.append(perm)

    partial_orders = [[0 for y in range(num_node)] for x in range(num_node)]
    for i in range(num_node):
        stabilized_aut = []
        for perm in valid_perms:
            if perm[i]==i:
                stabilized_aut.append(perm)
            else:
                partial_orders[perm[i]][i]=1

        valid_perms = stabilized_aut
    

    res = [-1 for x in range(num_node)]
    for i in range(num_node):
        largest_idx = -1
        for j in range(num_node):
            if partial_orders[i][j]==1 and j>largest_idx:
                largest_idx = j
        
        res[i] = largest_idx
    print("partial orders:")
    print(res)
    return res

# def read_graph(partial_edge):
#     np_rowptr = np.fromfile("../citeseer/snap.txt.vertex.bin", dtype=np.int64)
#     np_colidx = np.fromfile("../citeseer/snap.txt.edge.bin", dtype=np.int32)

#     torch_rowptr = torch.from_numpy(np_rowptr, ).to(torch.int32)
#     torch_colidx =torch.from_numpy(np_colidx)
#     torch_edge_list = torch.zeros([torch_colidx.shape[0], 2], dtype=torch.int32)

#     edge_idx = 0
#     for i in range(0, torch_rowptr.shape[0]-1):
#         for j in range(torch_rowptr[i].item() , torch_rowptr[i+1].item()):
#                 if (not partial_edge) or (partial_edge and torch_colidx[j]<i):
#                     torch_edge_list[edge_idx][0] = i
#                     torch_edge_list[edge_idx][1] = torch_colidx[j]
#                     edge_idx = edge_idx+1
#     #Cuke
#     num_node = torch_rowptr.shape[0]-1
#     num_edges = torch_colidx.shape[0]
#     if partial_edge:
#         num_jobs = edge_idx
#     else:
#         num_jobs = num_edges

#     return torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs


# def read_pattern_file(filename):
#     pmtx = None
#     num_node=0
#     with open(filename) as p:
#         for line in p:
#             if line[0]=='v':
#                 num_node+=1
#         p.seek(0, 0)

#         pmtx = [[0 for y in range(num_node)] for x in range(num_node)]
#         for line in p: 
#             if line[0]=='e':
#                 v0 = int(line[2])
#                 v1 = int(line[4])
#                 pmtx[v0][v1] = 1
#                 pmtx[v1][v0] = 1
#         print("pmtx:")
#         print(pmtx)
#         return pmtx



# def node_degree(pmtx, idx):
#     res = 0
#     row = pmtx[idx]
#     for i in row:
#         if i==1:
#             res+=1
#     return res

# def read_query_file(filename):
#     pmtx = None
#     num_node=0
#     with open(filename) as p:
#         for line in p:
#             if line[0]=='v':
#                 num_node+=1
#         p.seek(0, 0)

#         pmtx = [[0 for y in range(num_node)] for x in range(num_node)]
#         for line in p: 
#             if line[0]=='e':
#                 v0 = int(line[2])
#                 v1 = int(line[4])
#                 pmtx[v0][v1] = 1
#                 pmtx[v1][v0] = 1
#         # print("pmtx:")
#         # print(pmtx)
#         return pmtx

# def query_info(pmtx):
#     num_edge = 0
#     num_node = len(pmtx)
#     edge_list = []
#     for row_idx in range(0, len(pmtx)):
#         for col_idx in range(0, len(pmtx[row_idx])):
#             item = pmtx[row_idx][col_idx]
#             if item==1 and row_idx<col_idx:
#                 edge_list.append([row_idx, col_idx])
#                 num_edge+=1
    
#     return num_node, num_edge, edge_list
    

# def query_dfs(pmtx):
#     edge_order = []

#     def _query_dfs(query_adjlist, cur_node, visited):
#         if cur_node not in visited:
#             visited.add(cur_node)
#             for adj_node in query_adjlist[cur_node]:
#                 if [cur_node, adj_node] not in edge_order and [adj_node, cur_node] not in edge_order:
#                     edge_order.append([cur_node, adj_node])
#                 _query_dfs(query_adjlist, adj_node, visited)
    
#     query_adjlist = []

#     for row_idx in range(0, len(pmtx)):
#         this_list = []
#         for col_idx in range(0, len(pmtx[row_idx])):
#             if pmtx[row_idx][col_idx]==1:
#                 this_list.append(col_idx)
        
#         query_adjlist.append(this_list)

#     _query_dfs(query_adjlist, 0, set())
#     return edge_order

# def node_degree(pmtx, idx):
#     res = 0
#     row = pmtx[idx]
#     for i in row:
#         if i==1:
#             res+=1
#     return res

# def fill_hash_table(node_list, rowptr, table):
#     src = inspect.cleandoc("""
#     for(int i=0; i<NODE_LIST_SIZE; i++){
#         int vid = NODE_LIST_ARRAY[i];
#         int hash_key = Hashing(HASH_TABLE, vid);
#         HASH_TABLE[hash_key][0] = NODE_LIST_ARRAY[i];
#         HASH_TABLE[hash_key][1] = ROWPTR[vid];
#         HASH_TABLE[hash_key][2] = ROWPTR[vid+1];
#     }
#     """)

#     table = inline(src, ('HASH_TABLE', table), \
#                         ('NODE_LIST_ARRAY', node_list), ('NODE_LIST_SIZE', node_list._size()[0]), \
#                         ('ROWPTR', rowptr))
#     return table