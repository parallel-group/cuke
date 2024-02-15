import inspect
from asg import *

def intersect(a, b):
    src = inspect.cleandoc("""
    RES_SIZE = SetIntersection(FIRST_TENSOR, FIRST_SIZE, SECOND_TENSOR, SECOND_SIZE, RES_TENSOR);
    """)
    output_size = Var(dtype='int')
    output_size.attr['is_arg'] = False

    output_tensor = Tensor((4096, ), dtype='int')
    output_tensor.attr['is_arg'] = False

    output_size = inline(src, ('RES_SIZE', output_size), \
                                ('FIRST_TENSOR', a[0]), ('FIRST_SIZE', a._size()[0]), \
                                ('SECOND_TENSOR', b[0]), ('SECOND_SIZE',  b._size()[0]),  \
                                ('RES_TENSOR', output_tensor[0]))
    ret_val = output_tensor[0:output_size]
    ret_val.ref_size[0].attr['dynamic_size'] = True
    ret_val.attr['is_set'] = True
    return ret_val


def difference(a, b):
    src = inspect.cleandoc("""
    RES_SIZE = SetDifference(FIRST_TENSOR, FIRST_SIZE, SECOND_TENSOR, SECOND_SIZE, RES_TENSOR);
    """)
    output_size = Var(dtype='int')
    output_size.attr['is_arg'] = False

    output_tensor = Tensor((4096, ), dtype='int')
    output_tensor.attr['is_arg'] = False

    output_size = inline(src, ('RES_SIZE', output_size), \
                                ('FIRST_TENSOR', a[0]), ('FIRST_SIZE', a._size()[0]), \
                                ('SECOND_TENSOR', b[0]), ('SECOND_SIZE',  b._size()[0]),  \
                                ('RES_TENSOR', output_tensor[0]))

    ret_val = output_tensor[0:output_size]
    ret_val.ref_size[0].attr['dynamic_size'] = True
    ret_val.attr['is_set'] = True
    return ret_val

# def intersect(a, b):
#     src = inspect.cleandoc("""
#     RES_SIZE = SetIntersection(FIRST_TENSOR, FIRST_SIZE, SECOND_TENSOR, SECOND_SIZE, RES_TENSOR);
#     """)
#     output_size = Var(dtype='int')
#     output_size.attr['is_arg'] = False

#     output_tensor = Tensor((4096, ), dtype='int')
#     output_tensor.attr['is_arg'] = False

#     output_tensor = inline(src, ('RES_TENSOR', output_tensor[0]),  \
#                                 ('FIRST_TENSOR', a[0]), ('FIRST_SIZE', a.size(0)), \
#                                 ('SECOND_TENSOR', b[0]), ('SECOND_SIZE',  b.size(0)),  \
#                                 ('RES_SIZE', output_size))
#     ret_val = output_tensor[0:output_size]
    
#     ret_val.ref_size[0].attr['dynamic_size'] = True
#     ret_val.attr['is_set'] = True
#     return ret_val

# def difference(a, b):
#     src = inspect.cleandoc("""
#     RES_SIZE = SetDifference(FIRST_TENSOR, FIRST_SIZE, SECOND_TENSOR, SECOND_SIZE, RES_TENSOR);
#     """)
#     output_size = Var(dtype='int')
#     output_size.attr['is_arg'] = False

#     output_tensor = Tensor((4096, ), dtype='int')
#     output_tensor.attr['is_arg'] = False

#     output_tensor = inline(src, ('RES_TENSOR', output_tensor[0]),  \
#                                 ('FIRST_TENSOR', a[0]), ('FIRST_SIZE', a._size()[0]), \
#                                 ('SECOND_TENSOR', b[0]), ('SECOND_SIZE',  b._size()[0]),  \
#                                 ('RES_SIZE', output_size))

#     ret_val = output_tensor[0:output_size]
#     ret_val.ref_size[0].attr['dynamic_size'] = True
#     ret_val.attr['is_set'] = True
#     return ret_val

# def intersect(a, b):
#     return TensorOp('intersection', a, b)

# def difference(a, b):
#     return TensorOp('difference', a, b)

# def is_in(x, li):
#     src = inspect.cleandoc("""
#     F = BinarySearch(LI, 0, LSIZE, X);
#     """)
#     found = Var(dtype='int')
#     found.attr['is_arg'] = False
#     return inline(src, ('F', found), ('X', x), ('LI', li[0]), ('LSIZE', li._size()[0]))

# def is_not_in(x, li):
#     src = inspect.cleandoc("""
#     F = !BinarySearch(LI, 0, LSIZE, X);
#     """)
#     found = Var(dtype='int')
#     found.attr['is_arg'] = False
#     return inline(src, ('F', found), ('X', x), ('LI', li[0]), ('LSIZE', li._size()[0]))


# def merge_is_in(x, second, pj):
#     src = inspect.cleandoc("""
#         if(PJ < SECOND_SIZE && X < SECOND_ONE){F = false; }\n\
#         else if (PJ < SECOND_SIZE && X > SECOND_ONE) { \n\
#             while(PJ < SECOND_SIZE && X > SECOND_ONE){    \n\
#                 PJ++;                                       \n\
#             } \n\
#         } \n\
#         if(PJ == SECOND_SIZE) {F = false; }\n\
#         if(PJ < SECOND_SIZE && X==SECOND_ONE){
#             F = true;
#         }
#         else{
#             F = false;
#         }
#     """)

#     found = Var(dtype='int')
#     found.attr['is_arg'] = False
#     return inline(src, ('F', found), ('X', x), ('SECOND_ONE', second[pj]), ('PJ', pj), ('SECOND_SIZE', second._size()[0]))


# def merge_is_not_in(x, second, pj):
#     src = inspect.cleandoc("""
#         if(PJ < SECOND_SIZE && X < SECOND_ONE){F = false; }\n\
#         else if (PJ < SECOND_SIZE && X > SECOND_ONE) { \n\
#             while(PJ < SECOND_SIZE && X > SECOND_ONE){    \n\
#                 PJ++;                                       \n\
#             } \n\
#         } \n\
#         if(PJ == SECOND_SIZE) {F = false; }\n\
#         if(PJ < SECOND_SIZE && X==SECOND_ONE){
#             F = false;
#         }
#         else{
#             F = true;
#         }
#     """)

#     found = Var(dtype='int')
#     found.attr['is_arg'] = False
#     return inline(src, ('F', found), ('X', x), ('SECOND_ONE', second[pj]), ('PJ', pj), ('SECOND_SIZE', second._size()[0]))

# def intersect(a, b):
#     c = a.apply(lambda x: is_in(x, b))
#     return a.apply(lambda x: x, cond=c)

# def difference(a, b):
#     c = a.apply(lambda x: is_not_in(x, b))
#     return a.apply(lambda x: x, cond=c)
