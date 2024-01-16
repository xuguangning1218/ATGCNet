import torch
import torch.nn as nn
from model.ATGCO import ATGCO, TensorProduct

class TensorGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relation, bias=False):
        super(TensorGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.num_relation = num_relation
        
        self.conv_xz = ATGCO(in_features=input_dim, out_features=hidden_dim, num_relation=self.num_relation)
        self.conv_xr = ATGCO(in_features=input_dim, out_features=hidden_dim, num_relation=self.num_relation)
        self.conv_xh = ATGCO(in_features=input_dim, out_features=hidden_dim, num_relation=self.num_relation)
        
        self.conv_hz = ATGCO(in_features=hidden_dim, out_features=hidden_dim, num_relation=self.num_relation)
        self.conv_hr = ATGCO(in_features=hidden_dim, out_features=hidden_dim, num_relation=self.num_relation)
        self.conv_hh = ATGCO(in_features=hidden_dim, out_features=hidden_dim, num_relation=self.num_relation)
                
        if self.bias == True:
            self.B = nn.Parameter(torch.Tensor(3, hidden_dim, num_relation))
            torch.nn.init.kaiming_normal_(self.B)
    
    def forward(self, X, adj, h_pre):
        
        if self.bias == True:
            self.B = self.B.to(X.device)
        
        b_z, b_r, b_h = (self.B[0], self.B[1], self.B[2]) if self.bias == True else (0, 0, 0)
        
        H = h_pre
        Z = torch.sigmoid(self.conv_xz(X, adj) + self.conv_hz(H, adj) + b_z)
        R = torch.sigmoid(self.conv_xr(X, adj) + self.conv_hr(H, adj) + b_r)
        H_tilda = torch.tanh(self.conv_xh(X, adj) + self.conv_hr(R * H, adj) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        return H

class ATG_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relation, num_node, embeded_dim, num_layers=2, bias=True):
        super(ATG_GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.num_node = num_node
        self.embeded_dim = embeded_dim
        self.U = nn.Parameter(torch.Tensor(self.num_node, self.embeded_dim, num_relation), requires_grad=True)
        nn.init.kaiming_normal_(self.U)
        
        self.cells = nn.ModuleList()
        self.tproduct = TensorProduct()
        
        self.linear =  nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_relation, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        
        for i in range(self.num_layers):
            if i == 0:
                self.cells.append(TensorGRUCell(input_dim=self.input_dim, hidden_dim=self.hidden_dim, bias=self.bias, num_relation=self.num_relation))
            else:
                self.cells.append(TensorGRUCell(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, bias=self.bias, num_relation=self.num_relation))
    
    def init_state(self, num_layers, batch_size, num_nodes, hidden_dim, num_relation, device):
        return torch.zeros((num_layers, batch_size, num_nodes, hidden_dim, num_relation), device=device)
    
    def forward(self, inputs, h_pre=None):
        
        adj = torch.softmax(torch.relu(self.tproduct(self.U, self.U.permute(1, 0, 2).contiguous())), dim=1)
        
        batch_size, seq_time, num_nodes, feature_in, num_relation = inputs.size()
        
        if h_pre == None:
            h_pre = self.init_state(
                num_layers=self.num_layers,
                batch_size=batch_size,
                num_nodes=num_nodes, 
                hidden_dim=self.hidden_dim, 
                num_relation=self.num_relation, 
                device=inputs.device)
        
        outputs = []
        
        temp_out = [[] for _ in range(seq_time)]
        for t in range(seq_time):
            X = inputs[:, t]
            for i in range(self.num_layers):
                if i == 0 and t == 0:
                    out = self.cells[i](X, adj, h_pre[i])
                elif i !=0 and t == 0:
                    out = self.cells[i](out, adj, h_pre[i])
                elif i ==0 and t != 0:
                    out = self.cells[i](X, adj, temp_out[t-1][i])
                elif i !=0 and t != 0:
                    out = self.cells[i](temp_out[t-1][i-1], adj, temp_out[t-1][i])
                temp_out[t].append(out)
            outputs.append(out)
            
        out = torch.stack(outputs, dim=1)
        return out, torch.stack(temp_out[-1])