import torch
import torch.nn as nn
import math
import tensorly
tensorly.set_backend('pytorch')

# faster tensor product based on FFT
class TensorProduct(torch.nn.Module):
    def __init__(self, ):
        super(TensorProduct, self).__init__()
    
    def forward(self, A, B):
        if len(A.shape) == 3: # without batch
            n1, n2, n3 = A.shape
        elif len(A.shape) == 4: # with batch
            b, n1, n2, n3 = A.shape
            
        if len(B.shape) == 3:  
            n2, n, n3 = B.shape
        elif len(B.shape) == 4: # with batch
            b, n2, n, n3 = B.shape
            
        A_transformed = torch.fft.fft(input=A, dim=-1)
        B_transformed = torch.fft.fft(input=B, dim=-1)
        
        C_transformed = []
        
        for k in range(n3):
            a_slice = A_transformed[...,k]
            b_slice = B_transformed[...,k]
            C_transformed.append(torch.matmul(a_slice,b_slice))
        C = torch.fft.ifft(input = torch.stack(C_transformed, dim=-1), dim=-1)
        return torch.real(C)

# version 2
class ATGCO(torch.nn.Module):
    """
    in_features: input dimensions
    out_features: out_features dimensions
    num_relation: number of relations
    """
    def __init__(self, in_features, out_features, num_relation, bias=False):
        super(ATGCO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relation = num_relation
        self.weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.tproduct = TensorProduct()
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        if x.device != self.weight.device:
            self.weight.to(x.device)
        # conduct t-product
        support = self.tproduct(adj, x)
        # conduct m-product
        output = tensorly.tenalg.mode_dot(tensor=support, matrix_or_vector=self.weight, mode=2, transpose=True)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
# version 1
class ATGCO_V1(torch.nn.Module):
    """
    in_features: input dimensions
    out_features: out_features dimensions
    num_relation: number of relations
    """
    def __init__(self, in_features, out_features, num_relation, bias=False):
        super(ATGCO_V1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relation = num_relation
        self.weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features, self.num_relation))
        self.tproduct = TensorProduct()
        # generate circle index
        self.circ = self.circ_generator(num_relation)
        # generate normal index
        self.index = self.index_generator(num_relation)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def circ_generator(self, N):
        A = np.zeros((N, N))
        row = np.arange(N)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i][j] = row[j]
            row = np.roll(row, 1)
        return A.T.astype(np.int32)
    
    def index_generator(self, N):
        A = np.zeros((N, N))
        row = np.arange(N)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i][j] = row[j]
        return A.astype(np.int32)
    
    def forward(self, x, adj):
        if x.device != self.weight.device:
            self.weight.to(x.device)
            
        output = []
        for i in range(num_relation):
            result = 0
            for j in range(num_relation):
                # get index
                _i, _j = self.index[i, j], self.circ[i,j]
                # conduct integration
                result += adj[..., _j] @ x[..., _i] @ self.weight[..., _j]
            if self.bias is not None:
                result += self.bias
            output.append(result)
        output = torch.stack(output, dim=-1)
        return output