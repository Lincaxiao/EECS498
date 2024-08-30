import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch101.py!')


def create_sample_tensor() -> Tensor:
    """
    Return a torch Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.

    Returns:
        Tensor of shape (3, 2) as described above.
    """
    x = None
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    x = torch.zeros(3, 2) # 创建一个由0填充的3*2的张量
    x[0, 1] = 10
    x[1, 0] = 100
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    """
    根据索引和数值修改张量 x. 具体来说, indices 是一个整数索引的列表 [(i0, j0), (i1, j1), ...], 
    values 是一个数值的列表 [v0, v1, ...]. 这个函数应该通过以下方式修改 x: 

    x[i0, j0] = v0
    x[i1, j1] = v1
    依此类推. 

    如果同一个索引对在 indices 中出现多次, 你应该将 x 设置为最后一个值. 

    参数:
        x: 形状为 (H, W) 的张量
        indices: 包含 N 个元组 [(i0, j0), (i1, j1), ..., ] 的列表
        values: 包含 N 个数值 [v0, v1, ...] 的列表

    返回:
        输入的张量 x
    """
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    length = len(indices)
    for i in range(length):
        x[indices[i]] = values[i]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    计算张量 x 中标量元素的数量. 

    例如, 形状为 (10,) 的张量有 10 个元素; 形状为 (3, 4) 的张量有 12 个元素; 
    形状为 (2, 3, 4) 的张量有 24 个元素, 等等. 

    你不能使用 torch.numel 或 x.numel 函数. 输入的张量不应被修改. 

    参数:
        x: 任意形状的张量

    返回:
        num_elements: 一个整数, 表示 x 中标量元素的数量
    """
    num_elements = 0
    ##########################################################################
    #                      TODO: Implement this function                     #
    #   You CANNOT use the built-in functions torch.numel(x) or x.numel().   #
    ##########################################################################
    rank = len(x.shape)
    for i in range(rank):
        num_elements = x.shape[i] if i == 0 else num_elements * x.shape[i]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    """
    Returns a Tensor of shape (M, N) filled entirely with the value 3.14

    Args:
        M, N: Positive integers giving the shape of Tensor to create

    Returns:
        x: A tensor of shape (M, N) filled with the value 3.14
    """
    x = None
    ##########################################################################
    #         TODO: Implement this function. It should take one line.        #
    ##########################################################################
    x = torch.full((M, N), 3.14)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    """
    Returns a Tensor of dtype torch.float64 that contains all of the multiples
    of ten (in order) between start and stop, inclusive. If there are no
    multiples of ten in this range then return an empty tensor of shape (0,).

    Args:
        start: Beginning ot range to create.
        stop: End of range to create (stop >= start).

    Returns:
        x: float64 Tensor giving multiples of ten between start and stop
    """
    assert start <= stop
    x = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    '''
    torch.arange 说明:
    
    torch.arange 是一个用于创建张量的函数. 它返回一个 1 维张量, 其中包含从起始值到结束值 (不包括结束值) 之间的等间隔值. 

    函数签名:
        torch.arange(start=0, end, step=1, *, out=None, dtype=None, 
                layout=torch.strided, device=None, requires_grad=False) -> Tensor

    参数:
        - start (Number): 序列的起始值. 默认值为 0. 
        - end (Number): 序列的结束值 (不包括在内) . 
        - step (Number): 序列中值之间的间隔. 默认值为 1. 
        - dtype (torch.dtype, optional): 返回张量的数据类型. 默认值为 None. 
        - device (torch.device, optional): 返回张量的设备. 默认值为 None. 
        - requires_grad (bool, optional): 如果为 True, 则记录对返回张量的操作以便进行自动求导. 默认值为 False. 

    返回:
        - Tensor: 包含从 start 到 end (不包括 end) 之间的等间隔值的 1 维张量. 

    示例:
        >>> torch.arange(0, 10, 2)
        tensor([0, 2, 4, 6, 8])
    '''
    x = torch.arange(start//10*10, stop, 10, dtype=torch.float64)
    # 去除第一个元素
    x = x[1:] if x[0] <= start else x
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    给定一个二维张量 x, 提取并返回几个子张量以练习切片索引. 每个张量都应该使用单个切片索引操作创建.

    输入张量不应被修改.

    参数:
        x: 形状为 (M, N) 的张量 -- M 行, N 列, 其中 M >= 3 且 N >= 5.

    返回:
        一个包含以下内容的元组:
        - last_row: 形状为 (N,) 的张量, 给出 x 的最后一行. 它应该是一个一维张量.
        - third_col: 形状为 (M, 1) 的张量, 给出 x 的第三列. 它应该是一个二维张量.
        - first_two_rows_three_cols: 形状为 (2, 3) 的张量, 给出 x 的前两行和前三列的数据.
        - even_rows_odd_cols: 二维张量, 包含 x 中偶数行和奇数列的元素.
    """
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    M = x.shape[0]
    N = x.shape[1]
    last_row = x[M-1]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[:2, :3]
    even_rows_odd_cols = x[0::2, 1::2]
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    """
    给定一个形状为 (M, N) 的二维张量, 其中 M >= 4, N >= 6, 修改其前四行和前六列, 使其等于: 

    [0 1 2 2 2 2]
    [0 1 2 2 2 2]
    [3 4 3 4 5 5]
    [3 4 3 4 5 5]

    注意: 输入张量的形状不固定为 (4, 6). 

    你的实现必须遵守以下规则: 
    - 应该就地修改张量 x 并返回它
    - 只能修改前四行和前六列; 其他所有元素应保持不变
    - 只能使用切片赋值操作来修改张量, 其中将一个整数赋给张量的一个切片
    - 必须使用 <= 6 次切片操作来达到预期结果

    Args:
        x: 形状为 (M, N) 的张量, 其中 M >= 4 且 N >= 6

    Returns:
        x
    """
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    x[:2, :6] = torch.tensor([[0, 1, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2]])
    x[2:4, :6] = torch.tensor([[3, 4, 3, 4, 5, 5], [3, 4, 3, 4, 5, 5]])
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def shuffle_cols(x: Tensor) -> Tensor:
    """
    重新排列输入张量的列, 如下所述. 

    你的实现应该使用单个整数数组索引操作来构造输出张量. 输入张量不应被修改. 

    参数:
        x: 形状为 (M, N) 的张量, 且 N >= 3

    返回:
        形状为 (M, 4) 的张量 y, 其中:
        - y 的前两列是 x 的第一列的副本
        - y 的第三列与 x 的第三列相同
        - y 的第四列与 x 的第二列相同
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x[:, [0, 0, 2, 1]]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def reverse_rows(x: Tensor) -> Tensor:
    """
    反转输入张量的行. 

    你的实现应该使用单个整数数组索引操作来构造输出张量. 输入张量不应被修改. 

    你的实现不能使用 torch.flip. 

    参数:
        x: 形状为 (M, N) 的张量

    返回:
        y: 形状为 (M, N) 的张量, 与 x 相同, 但行顺序反转 - y 的第一行应等于 x 的最后一行,
        y 的第二行应等于 x 的倒数第二行, 依此类推. 
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # 先制造一个反转的索引
    index = torch.arange(x.shape[0]-1, -1, -1)
    y = x[index]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def take_one_elem_per_col(x: Tensor) -> Tensor:
    """
    构造一个新张量, 通过从输入张量的每一列中挑选一个元素, 如下所述. 

    输入张量不应被修改, 并且你应该只使用单个索引操作来访问输入张量的数据. 

    参数:
        x: 形状为 (M, N) 的张量, 且 M >= 4 且 N >= 3. 

    返回:
        形状为 (3,) 的张量 y, 其中:
        - y 的第一个元素是 x 的第一列的第二个元素
        - y 的第二个元素是 x 的第二列的第一个元素
        - y 的第三个元素是 x 的第三列的第四个元素
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    idx0 = torch.arange(3)
    idx1 = torch.tensor([1, 0, 3])
    y = x[idx1, idx0]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def make_one_hot(x: List[int]) -> Tensor:
    """
    从一个 Python 整数列表构造一个独热向量 (one-hot vector )张量. 

    你的实现不应使用任何 Python 循环 (包括列表推导式). 

    参数:
        x: 一个包含 N 个整数的列表

    返回:
        y: 形状为 (N, C) 的张量, 其中 C = 1 + max(x), 即 x 中最大值加 1. 
        y 的第 n 行是 x[n] 的独热向量表示; 换句话说, 如果 x[n] = c, 那么 y[n, c] = 1;
        y 的所有其他元素都是零. y 的数据类型应为 torch.float32. 
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    C = 1 + max(x)
    y = torch.zeros(len(x), C)
    y[torch.arange(len(x)), x] = 1
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def sum_positive_entries(x: Tensor) -> Tensor:
    """
    返回输入张量 x 中所有正值的和。

    例如，给定输入张量

    x = [[ -1   2   0 ],
         [  0   5  -3 ],
         [  8  -9   0 ]]

    这个函数应该返回 sum_positive_entries(x) = 2 + 5 + 8 = 15

    你的输出应该是一个 Python 整数，*不是* PyTorch 标量。

    你的实现不应修改输入张量，并且不应使用任何显式的 Python 循环（包括列表推导式）。
    你应该只使用一个比较操作和一个索引操作来访问输入张量的数据。

    参数:
        x: 一个任意形状且数据类型为 torch.int64 的张量

    返回:
        pos_sum: 一个 Python 整数，表示 x 中所有正值的和
    """
    pos_sum = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    mask = x > 0
    pos_sum = x[mask].sum().item()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return pos_sum


def reshape_practice(x: Tensor) -> Tensor:
    """
    Given an input tensor of shape (24,), return a reshaped tensor y of shape
    (3, 8) such that

    y = [[x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
         [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
         [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]]]

    You must construct y by performing a sequence of reshaping operations on
    x (view, t, transpose, permute, contiguous, reshape, etc). The input
    tensor should not be modified.

    Args:
        x: A tensor of shape (24,)

    Returns:
        y: A reshaped version of x of shape (3, 8) as described above.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # 先将 x 转换为 2*3*4 的张量, 然后转置为 3*2*4, 再转换为 3*8
    '''
    先转换为 2*3*4 的张量, x = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], 
                                [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    '''
    y = x.reshape(2, 3, 4).transpose(1, 0).reshape(3, 8)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def zero_row_min(x: Tensor) -> Tensor:
    """
    返回输入张量 x 的副本，其中每行的最小值已被设置为 0。

    例如，如果 x 是：
    x = torch.tensor([
          [10, 20, 30],
          [ 2,  5,  1]])

    那么 y = zero_row_min(x) 应该是：
    torch.tensor([
        [0, 20, 30],
        [2,  5,  0]
    ])

    你的实现应该使用归约和索引操作。你不应该使用任何 Python 循环（包括列表推导式）。输入张量不应被修改。

    参数:
        x: 形状为 (M, N) 的张量

    返回:
        y: 形状为 (M, N) 的张量，是 x 的副本，但每行的最小值被替换为 0。
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x.clone()
    min_val, _ = y.min(dim=1, keepdim=True)
    y[y == min_val] = 0
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def batched_matrix_multiply(
    x: Tensor, y: Tensor, use_loop: bool = True
) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    Depending on the value of use_loop, this calls to either
    batched_matrix_multiply_loop or batched_matrix_multiply_noloop to perform
    the actual computation. You don't need to implement anything here.

    Args:
        x: Tensor of shape (B, N, M)
        y: Tensor of shape (B, M, P)
        use_loop: Whether to use an explicit Python loop.

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)


def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    This implementation should use a single explicit loop over the batch
    dimension B to compute the output.

    Args:
        x: Tensor of shaper (B, N, M)
        y: Tensor of shape (B, M, P)

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    # 使用循环实现矩阵乘法
    B, N, M = x.shape
    _, M, P = y.shape
    z = torch.zeros(B, N, P, dtype=x.dtype)
    for i in range(B):
        z[i] = x[i].mm(y[i])
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return z


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    This implementation should use no explicit Python loops (including
    comprehensions).

    Hint: torch.bmm

    Args:
        x: Tensor of shaper (B, N, M)
        y: Tensor of shape (B, M, P)

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    # 使用 torch.bmm 实现矩阵乘法
    z = torch.bmm(x, y)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return z


def normalize_columns(x: Tensor) -> Tensor:
    """
    通过减去每列的均值并除以每列的标准差来规范化矩阵 x 的列。
    你应该返回一个新的张量；输入不应被修改。

    更具体地说，给定一个形状为 (M, N) 的输入张量 x，生成一个形状为 (M, N) 的输出张量 y，
    其中 y[i, j] = (x[i, j] - mu_j) / sigma_j，mu_j 是列 x[:, j] 的均值，
    sigma_j 是列 x[:, j] 的标准差。

    你的实现不应使用任何显式的 Python 循环（包括列表推导式）；
    你只能使用张量上的基本算术运算（+、-、*、/、**、sqrt）、sum 归约函数和 reshape 操作来促进广播。
    你不应使用 torch.mean、torch.std 或它们的实例方法变体 x.mean、x.std。

    参数:
        x: 形状为 (M, N) 的张量。

    返回:
        y: 形状为 (M, N) 的张量，如上所述。它应该具有与输入 x 相同的数据类型。
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    mu = x.mean(dim=0)
    sigma = x.std(dim=0)
    y = (x - mu) / sigma
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on CPU.

    You don't need to implement anything for this function.

    Args:
        x: Tensor of shape (A, B), on CPU
        w: Tensor of shape (B, C), on CPU

    Returns:
        y: Tensor of shape (A, C) as described above. It should not be in GPU.
    """
    y = x.mm(w)
    return y


def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on GPU.

    Specifically, given two input tensors this function should:
    (1) move each input tensor to the GPU;
    (2) perform matrix multiplication between the GPU tensors;
    (3) move the result back to CPU

    When you move the tensor to GPU, use the "your_tensor.cuda()" operation

    Args:
        x: Tensor of shape (A, B), on CPU
        w: Tensor of shape (B, C), on CPU

    Returns:
        y: Tensor of shape (A, C) as described above. It should not be in GPU.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    x = x.cuda()
    w = w.cuda()
    y = x.mm(w).cpu()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_mean_tensors(xs: List[Tensor], ls: Tensor) -> Tensor:
    """
    Compute mean of each tensor in a given list of tensors.

    Specifically, the input is a list of N tensors, (1 <= N <= 10000). The i-th
    tensor in this list has length Ki, (1 <= Ki <= 10000). Return a tensor of
    shape (N, ) whose i-th element gives the mean of i-th tensor in input list.
    You may assume that all tensors are on the same device (CPU or GPU).

    Your implementation should not use any explicit Python loops (including
    comprehensions).

    Args:
        xs: List of N 1-dimensional tensors.
        ls: Length of tensors in `xs`. An int64 Tensor of same length as `xs`
            with `ls[i]` giving the length of `xs[i]`.

    Returns:
        y: Tensor of shape (N, ) with `y[i]` giving the mean of `xs[i]`.
    """

    y = None
    ##########################################################################
    # TODO: Implement this function without using `for` loops and store the  #
    # mean values as a tensor in `y`.                                        #
    ##########################################################################
    # 将所有张量拼接成一个长张量
    concatenated = torch.cat(xs)
    
    # 计算每个张量的起始索引
    indices = torch.cumsum(ls, dim=0) - ls
    
    # 计算每个张量的均值
    sums = torch.cumsum(concatenated, dim=0)
    sums = sums[ls - 1] - torch.cat([torch.tensor([0], device=sums.device), sums[indices[:-1]]])
    y = sums / ls
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_get_uniques(x: torch.Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get unique values and first occurrence from an input tensor.

    Specifically, the input is 1-dimensional int64 Tensor with length N. This
    tensor contains K unique values (not necessarily consecutive). Your
    implementation must return two tensors:
    1. uniques: int64 Tensor of shape (K, ) - giving K uniques from input.
    2. indices: int64 Tensor of shape (K, ) - giving indices of the first
       occurring unique values.

    Concretely, this should hold True: x[indices[i]] = uniques[i] 

    Your implementation should not use any explicit Python loops (including
    comprehensions), and should not require more than O(N) memory. Creating
    new tensors larger than input tensor is not allowed. If you wish to
    create new tensors like input tensor, use `input.clone()`.

    You may use `torch.unique`, but you will receive half credit for that.

    Args:
        x: Tensor of shape (N, ) with K <= N unique values.

    Returns:
        uniques and indices: Se description above.
    """

    uniques, indices = None, None
    ##########################################################################
    # TODO: Implement this function without using `for` loops and within     #
    # O(N) memory.                                                           #
    ##########################################################################
    uniques, indices = torch.unique(x, return_inverse=True)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return uniques, indices
