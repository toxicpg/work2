from torch import nn
import torch


# Static GCN w/ dense adj
class GCN(nn.Module):
    def __init__(self, K:int, input_dim:int, hidden_dim:int, bias=True, activation=nn.ReLU):
        super().__init__()
        self.K = K   # 支持矩阵个数
        self.input_dim = input_dim   #每个节点输入特征维度
        self.hidden_dim = hidden_dim  #输出隐藏维度
        self.bias = bias
        self.activation = activation() if activation is not None else None #实例化激活函数
        self.init_params(n_supports=K)


    # W 的形状： (n_supports * input_dim, hidden_dim)
    def init_params(self, n_supports:int, b_init=0):
        self.W = nn.Parameter(torch.empty(n_supports*self.input_dim, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)               # 创建一个未初始化的张量
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, A:torch.Tensor, x:torch.Tensor):
        '''
        Batch-wise graph convolution operation on given list of support adj matrices
        :param A: support adj matrices - torch.Tensor (K, n_nodes, n_nodes) 支持矩阵堆叠
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, hidden_dim)
        '''
        assert self.K == A.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum('ij,bjp->bip', [A[k,:,:], x]) #对每个支持矩阵 A[k]，做矩阵乘法 A[k] @ x
            support_list.append(support) # 形状：support 为 (B, N, input_dim)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum('bip,pq->biq', [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output



class Adj_Preprocessor(object):
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, adj:torch.Tensor):
        '''
        Generate adjacency matrices
        :param adj: input adj matrix - (N, N) torch.Tensor
        :return: processed adj matrix - (K_supports, N, N) torch.Tensor
        '''
        kernel_list = list()

        if self.kernel_type in ['localpool', 'chebyshev']:  # spectral
            adj_norm = self.symmetric_normalize(adj)
            # adj_norm = self.random_walk_normalize(adj)     # for asymmetric normalization
            if self.kernel_type == 'localpool':
                localpool = torch.eye(adj_norm.shape[0]) + adj_norm  # same as add self-loop first
                kernel_list.append(localpool)

            else:  # chebyshev
                laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                rescaled_laplacian = self.rescale_laplacian(laplacian_norm)
                kernel_list = self.compute_chebyshev_polynomials(rescaled_laplacian, kernel_list)

        elif self.kernel_type == 'random_walk_diffusion':  # spatial

            # diffuse k steps on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)
            '''
            # diffuse k steps bidirectionally on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            P_backward = self.random_walk_normalize(adj.T)
            forward_series, backward_series = [], []
            forward_series = self.compute_chebyshev_polynomials(P_forward.T, forward_series)
            backward_series = self.compute_chebyshev_polynomials(P_backward.T, backward_series)
            kernel_list += forward_series + backward_series[1:]  # 0-order Chebyshev polynomial is same: I
            '''
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')

        # print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
        kernels = torch.stack(kernel_list, dim=0)

        return kernels

    @staticmethod
    def random_walk_normalize(A):   # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)   # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.
        D = torch.diag(d_inv)
        A_norm = torch.mm(D, A)
        return A_norm

    @staticmethod
    def symmetric_normalize(A):
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = torch.mm(torch.mm(D, A), D)
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
        try:
            lambda_complex = torch.linalg.eigvals(L)
            lambda_ = lambda_complex.real  # 取实部      # get the real parts of eigenvalues
            lambda_max = lambda_.max()      # get the largest eigenvalue
        except:
            print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescale = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescale

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        return T_k