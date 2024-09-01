"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    计算每个训练集元素和每个测试集元素之间的平方欧几里得距离. 
    图像应展平并视为向量. 

    这个实现使用了对训练数据和测试数据的嵌套循环. 

    输入数据可以有任意维度——例如，这个函数应该能够计算向量之间的最近邻，
    在这种情况下，输入的形状将是 (num_{train, test}, D); 
    它也应该能够计算图像之间的最近邻，
    在这种情况下，输入的形状将是 (num_{train, test}, C, H, W). 
    更一般地，输入的形状将是 (num_{train, test}, D1, D2, ..., Dn); 
    你应该将形状为 (D1, D2, ..., Dn) 的每个元素展平成形状为
    (D1 * D2 * ... * Dn) 的向量，然后再计算距离. 

    输入张量不应被修改. 

    注意: 你的实现不能使用 `torch.norm`、`torch.dist`、
    `torch.cdist` 或它们的实例方法变体（`x.norm`、`x.dist`、
    `x.cdist` 等）. 你不能使用 `torch.nn` 或
    `torch.nn.functional` 模块中的任何函数. 

    参数:
        x_train: 形状为 (num_train, D1, D2, ...) 的张量
        x_test: 形状为 (num_test, D1, D2, ...) 的张量

    返回:
        dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j]
            是第 i 个训练点和第 j 个测试点之间的平方欧几里得距离. 
            它应该具有与 x_train 相同的数据类型. 
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]  # 获取训练集的数量
    num_test = x_test.shape[0]  # 获取测试集的数量
    # 初始化 dists 为形状为 (num_train, num_test) 的张量
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using a pair of nested loops over the    #
    # training data and the test data.                                       #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # 首先平面化 x_train 和 x_test
    x_train = x_train.view(num_train, -1)  # 即每个训练样本都被展平为一个向量
    x_test = x_test.view(num_test, -1)  # 即每个测试样本都被展平为一个向量
    # 计算欧式距离，使用显式循环
    for i in range(num_train):  # 遍历训练集
        for j in range(num_test):  # 遍历测试集
            # dists[i, j] 表示第 i 个训练样本和第 j 个测试样本之间的平方欧式距离
            dists[i, j] = torch.sum((x_train[i] - x_test[j]) ** 2)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    计算每个训练集元素和每个测试集元素之间的平方欧几里得距离。
    图像应展平并视为向量。

    这个实现仅使用一个对训练数据的循环。

    类似于 `compute_distances_two_loops`，这个函数应该能够处理
    任意维度的输入。输入数据不应被修改。

    注意：你的实现不能使用 `torch.norm`、`torch.dist`、
    `torch.cdist` 或它们的实例方法变体（`x.norm`、`x.dist`、
    `x.cdist` 等）。你不能使用 `torch.nn` 或
    `torch.nn.functional` 模块中的任何函数。

    参数:
        x_train: 形状为 (num_train, D1, D2, ...) 的张量
        x_test: 形状为 (num_test, D1, D2, ...) 的张量

    返回:
        dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j]
            是第 i 个训练点和第 j 个测试点之间的平方欧几里得距离。
            它应该具有与 x_train 相同的数据类型。
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using only a single loop over x_train.   #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    x_train = x_train.view(num_train, -1)
    x_test = x_test.view(num_test, -1)
    # 只使用一个循环
    for i in range(num_train):
        # dim = 1表示按行求和
        dists[i] = torch.sum((x_train[i] - x_test) ** 2, dim=1)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    计算每个训练集元素和每个测试集元素之间的平方欧几里得距离。
    图像应展平并视为向量。

    这个实现不应使用任何 Python 循环。为了节省内存，
    它也不应创建任何大的中间张量；特别是，
    你不应创建任何具有 O(num_train * num_test) 元素的中间张量。

    类似于 `compute_distances_two_loops`，这个实现应该能够处理
    任意维度的输入。输入数据不应被修改。

    注意：你的实现不能使用 `torch.norm`、`torch.dist`、
    `torch.cdist` 或它们的实例方法变体（`x.norm`、`x.dist`、
    `x.cdist` 等）。你不能使用 `torch.nn` 或
    `torch.nn.functional` 模块中的任何函数。

    参数:
        x_train: 形状为 (num_train, C, H, W) 的张量
        x_test: 形状为 (num_test, C, H, W) 的张量

    返回:
        dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j]
            是第 i 个训练点和第 j 个测试点之间的平方欧几里得距离。
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function without using any explicit loops and     #
    # without creating any intermediate tensors with O(num_train * num_test) #
    # elements.                                                              #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    #                                                                        #
    # HINT: Try to formulate the Euclidean distance using two broadcast sums #
    #       and a matrix multiply.                                           #
    ##########################################################################
    x_train = x_train.view(num_train, -1)
    x_test = x_test.view(num_test, -1)
    # 计算训练集和测试集的平方和
    train_squared = torch.sum(x_train ** 2, dim=1, keepdim=True)  # 形状为 (num_train, 1)
    test_squared = torch.sum(x_test ** 2, dim=1, keepdim=True)  # 形状为 (num_test, 1)

    # 计算交叉项
    cross_term = torch.mm(x_train, x_test.t())  # 形状为 (num_train, num_test)

    # 计算平方欧几里得距离，**广播**
    dists = train_squared - 2 * cross_term + test_squared.t()

    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    """
    给定所有训练样本和测试样本之间的距离，通过在训练集中对每个测试样本的 `k` 
    个最近邻进行多数投票来预测每个测试样本的标签. 

    如果出现平局，该函数应返回最小的标签. 例如，如果 k=5 且某个测试样本的 5 
    个最近邻的标签为 [1, 2, 1, 2, 3]，则 1 和 2 之间出现平局（每个都有 2 票），
    因此我们应该返回 1，因为它是最小的标签. 

    该函数不应修改其任何输入. 

    参数:
        dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j] 
        是第 i 个训练点和第 j 个测试点之间的平方欧几里得距离. 
        y_train: 形状为 (num_train,) 的张量，给出所有训练样本的标签. 
        每个标签都是范围 [0, num_classes - 1] 内的整数. 
        k: 用于分类的最近邻的数量. 

    返回:
        y_pred: 形状为 (num_test,) 的 int64 张量，给出测试数据的预测标签，
        其中 y_pred[j] 是第 j 个测试样本的预测标签. 每个标签应为
        范围 [0, num_classes - 1] 内的整数. 
    """
    num_train, num_test = dists.shape  # 获取训练集和测试集的数量
    y_pred = torch.zeros(num_test, dtype=torch.int64)  # 初始化预测标签
    ##########################################################################
    # TODO: Implement this function. You may use an explicit loop over the   #
    # test samples.                                                          #
    #                                                                        #
    # HINT: Look up the function torch.topk                                  #
    ##########################################################################
    y_train = y_train.view(-1, 1)  # 将 y_train 转换为列向量，y_train 的形状为 (num_train, 1)
    # 获取最近的 k 个邻居的索引, indices 的形状为 (k, num_test)
    _, indices = torch.topk(dists, k=k, dim=0, largest=False)  # 沿着列维度获取最小的 k 个值
    # 获取最近的 k 个邻居的标签
    k_labels = y_train[indices]
    # 计算每个测试样本的预测标签，不使用循环
    y_pred, _ = torch.mode(k_labels, dim=0)  # 沿着列维度计算众数
    y_pred = y_pred.view(-1)  # 将 y_pred 转换为行向量
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels
        """
        ######################################################################
        # TODO: Implement the initializer for this class. It should perform  #
        # no computation and simply memorize the training data in            #
        # `self.x_train` and `self.y_train`, accordingly.                    #
        ######################################################################
        self.x_train = x_train
        self.y_train = y_train
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################

    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        """
        y_test_pred = None
        ######################################################################
        # TODO: Implement this method. You should use the functions you      #
        # wrote above for computing distances (use the no-loop variant) and  #
        # to predict output labels.                                          #
        ######################################################################
        # 获取计算的距离
        dists = compute_distances_no_loops(self.x_train, x_test)
        # 预测标签
        y_test_pred = predict_labels(dists, self.y_train, k=k)
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################
        return y_test_pred

    def check_accuracy(
            self,
            x_test: torch.Tensor,
            y_test: torch.Tensor,
            k: int = 1,
            quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        num_folds: int = 5,
        k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    """
    Perform cross-validation for `KnnClassifier`.

    Args:
        x_train: Tensor of shape (num_train, C, H, W) giving all training data.
        y_train: int64 Tensor of shape (num_train,) giving labels for training
            data.
        num_folds: Integer giving the number of folds to use.
        k_choices: List of integers giving the values of k to try.

    Returns:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.
    """

    # First we divide the training data into num_folds equally-sized folds.
    x_train_folds = []
    y_train_folds = []
    ##########################################################################
    # TODO: 将训练数据和图像分成多个折叠。分割后，#
    # x_train_folds 和 y_train_folds 应该是长度为 num_folds 的列表，
    # 其中 y_train_folds[i] 是 x_train_folds[i] 中图像的标签向量。
    # 
    # 提示：使用 torch.chunk                                             #
    ##########################################################################
    # torch.chunk(tensor, chunks, dim=0) 沿着指定维度将 tensor 分割成 chunks 份
    x_train_folds = torch.chunk(x_train, num_folds, dim=0)  # 沿着行维度分割
    y_train_folds = torch.chunk(y_train, num_folds, dim=0)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    # A dictionary holding the accuracies for different values of k that we
    # find when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the
    # different accuracies we found trying `KnnClassifier`s using k neighbors.
    k_to_accuracies = {}

    ##########################################################################
    # TODO: 执行交叉验证以找到最佳的 k 值。对于 k_choices 中的每个 k 值，
    # 运行 k-NN 算法 num_folds 次；在每种情况下，你将使用除一个折叠外的所有折叠作为训练数据，
    # 并使用最后一个折叠作为验证集。将所有折叠和所有 k 值的准确率存储在 k_to_accuracies 中。
    #
    # 提示：使用 torch.cat
    ##########################################################################
    for k in k_choices:
        accuracies = []
        for i in range(num_folds):
            # 将除了第 i 个 fold 以外的所有 fold 拼接在一起
            x_train_fold = torch.cat(x_train_folds[:i] + x_train_folds[i + 1:], dim=0)
            y_train_fold = torch.cat(y_train_folds[:i] + y_train_folds[i + 1:], dim=0)
            # 创建 KnnClassifier 实例
            knn = KnnClassifier(x_train_fold, y_train_fold)
            # 计算准确率
            accuracy = knn.check_accuracy(x_train_folds[i], y_train_folds[i], k=k, quiet=True)
            accuracies.append(accuracy)
        k_to_accuracies[k] = accuracies
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    """
    Select the best value for k, from the cross-validation result from
    knn_cross_validate. If there are multiple k's available, then you SHOULD
    choose the smallest k among all possible answer.

    Args:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.

    Returns:
        best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info.
    """
    best_k = 0
    ##########################################################################
    # TODO: Use the results of cross-validation stored in k_to_accuracies to #
    # choose the value of k, and store result in `best_k`. You should choose #
    # the value of k that has the highest mean accuracy accross all folds.   #
    ##########################################################################
    # 选择平均准确率最高的 k
    best_k = max(k_to_accuracies, key=lambda k: sum(k_to_accuracies[k]) / len(k_to_accuracies[k]))
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return best_k
