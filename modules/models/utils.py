
def transpose_custom(tensor, dim0: int, dim1: int):
    # Ref to https://www.mindspore.cn/docs/zh-CN/r1.9/note/api_mapping/pytorch_diff/Tensor.transpose.html
    size = len(tensor.shape)
    dims = [i for i in range(size)]
    dims[dim0], dims[dim1] = dim1, dim0
    dims = tuple(dims)
    return tensor.transpose(dims)