"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


def hello_linear_classifier():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            learning_rate: float = 1e-3,
            reg: float = 1e-5,
            num_iters: int = 100,
            batch_size: int = 200,
            verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
            self,
            W: torch.Tensor,
            X_batch: torch.Tensor,
            y_batch: torch.Tensor,
            reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
            self,
            W: torch.Tensor,
            X_batch: torch.Tensor,
            y_batch: torch.Tensor,
            reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
            self,
            W: torch.Tensor,
            X_batch: torch.Tensor,
            y_batch: torch.Tensor,
            reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(
        W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    结构化 SVM 损失函数, 朴素实现 (带循环).

    输入的维度为 D, 有 C 个类别, 我们对 N 个样本的小批量进行操作.
    当你实现 W 的正则化时, 请不要将正则化项乘以 1/2 (没有系数 ).

    输入：
    - W: 一个形状为 (D, C) 的 PyTorch 张量, 包含权重. 
    - X: 一个形状为 (N, D) 的 PyTorch 张量, 包含一个小批量数据. 
    - y: 一个形状为 (N,) 的 PyTorch 张量, 包含训练标签; y[i] = c 表示
      X[i] 的标签为 c, 其中 0 <= c < C. 
    - reg: (float) 正则化强度

    返回一个元组：
    - 损失, 作为 torch 标量
    - 损失相对于权重 W 的梯度; 一个与 W 形状相同的张量
    """
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):  # 遍历所有训练样本
        # W.t()的形状为(C, D), X[i]的形状为(D,), 所以scores的形状为(C, 1)
        # x[i]的形状为(D,)的原因是PyTorch会自动将1维张量转换为2维张量
        scores = W.t().mv(X[i])  # scores = W^T * X, 计算所有类别的得分, scores的形状为(C,)
        correct_class_score = scores[y[i]]  # 正确类别的得分
        for j in range(num_classes):  # 遍历所有类别
            if j == y[i]:  # 跳过正确类别
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #######################################################################
                # Compute the gradient of the SVM term of the loss function and store #
                # it on dW. (part 1) Rather than first computing the loss and then    #
                # computing the derivative, it is simple to compute the derivative    #
                # at the same time that the loss is being computed.                   #
                #######################################################################
                # 计算导数, dW[:, j]表示错误类别的权重, dW[:, y[i]]表示正确类别的权重
                # 注意梯度下降需要的是负梯度，所以求梯度的时候是加上错误类别的权重，减去正确类别的权重
                dW[:, j] += X[i]  # 对错误类别的权重进行更新
                dW[:, y[i]] -= X[i]  # 对正确类别的权重进行更新
                #######################################################################
                #                       END OF YOUR CODE                              #
                #######################################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    #############################################################################
    # 计算梯度
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW


def svm_loss_vectorized(
        W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    loss = 0.0
    num_train = X.shape[0]
    # scores 形状为 (N, C), 其中 scores[i, j] 表示第 i 个训练样本的第 j 个类别的得分
    scores = X.mm(W)  # 计算所有类别的得分
    # correct_class_score 形状为 (N,), 其中 correct_class_score[i] 表示第 i 个训练样本的正确类别的得分
    # torch.arrange使得第一个参数为0,1,2,...,num_train-1
    correct_class_score = scores[torch.arange(num_train), y]
    # margin 形状为 (N, C), 其中 margin[i, j] 表示第 i 个训练样本的第 j 个类别的得分
    # 与正确类别的得分的差值加上1
    margin = scores - correct_class_score.view(-1, 1) + 1
    margin[torch.arange(num_train), y] = 0  # 将正确类别的位置置为0
    margin = torch.max(margin, torch.zeros_like(margin))  # 将小于0的位置置为0
    # 计算loss
    loss = torch.sum(torch.max(margin, torch.zeros_like(margin))) / num_train
    loss += reg * torch.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dW = torch.zeros_like(W)
    margin[margin > 0] = 1  # 将大于0的位置置为1, 原因是max(0, x)的导数为1 (x > 0)
    margin[torch.arange(num_train), y] = -torch.sum(margin, dim=1)  # 将正确类别的位置置为负梯度
    # 权重进行更新
    dW = X.t().mm(margin)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def sample_batch(
        X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in     #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
    # and y_batch should have shape (batch_size,)                           #
    #                                                                       #
    # Hint: Use torch.randint to generate indices.                          #
    #########################################################################
    # 使用 torch.randint 生成索引
    indices = torch.randint(num_train, (batch_size,))
    X_batch = X[indices]
    y_batch = y[indices]
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return X_batch, y_batch


def train_linear_classifier(
        loss_func: Callable,
        W: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # TODO: implement sample_batch function
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        #########################################################################
        # TODO:                                                                 #
        # Update the weights using the gradient and the learning rate.          #
        #########################################################################
        # 梯度下降 (减法是因为需要负梯度)
        W -= learning_rate * grad
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # 计算得分
    scores = X.mm(W)
    # 取得分最高的类别
    y_pred = torch.argmax(scores, dim=1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO:   add your own hyper parameter lists.                             #
    ###########################################################################
    learning_rates = [5e-3, 6e-3, 7e-3, 8e-3, 9e-3]
    regularization_strengths = [5e-2, 5.5e-2, 6e-2, 6.5e-2, 7e-2]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths


def test_one_param_set(
        cls: LinearClassifier,
        data_dict: Dict[str, torch.Tensor],
        lr: float,
        reg: float,
        num_iters: int = 4000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.
    ###########################################################################
    # TODO:                                                                   #
    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #
    #                                                                         #
    # Hint: Once you are confident that your validation code works, you       #
    # should rerun the validation code with the final value for num_iters.    #
    # Before that, please test with small num_iters first                     #
    ###########################################################################
    # 训练
    cls.train(data_dict["X_train"], data_dict["y_train"], lr, reg, num_iters)
    y_train_pred = cls.predict(data_dict["X_train"])
    y_val_pred = cls.predict(data_dict["X_val"])
    # 计算准确率
    train_acc = torch.mean((y_train_pred == data_dict["y_train"]).float())
    val_acc = torch.mean((y_val_pred == data_dict["y_val"]).float())
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
        W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, naive implementation (with loops).  When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(
        W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths
