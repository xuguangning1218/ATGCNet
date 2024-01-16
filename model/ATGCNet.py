import torch
import torch.nn as nn
from model.ATG_GRU import ATG_GRU

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_node, embeded_dim, num_relation, num_layers):
        super(Encoder, self).__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.num_node = num_node
        self.embeded_dim = embeded_dim
        self.gru = ATG_GRU(input_dim=in_features, 
                            hidden_dim=hidden_features, 
                            output_dim=out_features, 
                            num_relation=num_relation, 
                            num_layers=num_layers,
                            num_node=num_node,
                            embeded_dim=embeded_dim)

    def forward(self, x, h_pre=None):
        batch, timestep, num_node, in_feat, num_rel = x.size()
        if h_pre == None:
            h_pre = self.initState(batch, num_node, num_rel, device=x.device)
        out, h_pre = self.gru(x, h_pre)
        return out, h_pre

    def initState(self, batch_size, num_node, num_relation, device):
        return torch.zeros((self.num_layers, batch_size, num_node, self.hidden_features, num_relation), device=device)
    

class Decoder(nn.Module):
    def __init__(self, in_features, hidden_features, num_node, embeded_dim, num_relation, out_features, num_layers):
        super(Decoder, self).__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.out_features = out_features
        self.num_node = num_node
        self.embeded_dim = embeded_dim
        self.gru = ATG_GRU(input_dim=in_features, 
                            hidden_dim=hidden_features, 
                            output_dim=out_features, 
                            num_relation=num_relation, 
                            num_layers=num_layers,
                            num_node=num_node,
                            embeded_dim=embeded_dim)

    def forward(self, decode_input, h_pre):
        out, h_pre = self.gru(decode_input, h_pre)
        return out, h_pre


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class ATGCNet(nn.Module):

    def __init__(self, in_features=7, mlp_output_dim=8, hidden_features=32, out_features=1, num_relation=3, num_node=354, embeded_dim=64, num_layers=2):
        super(ATGCNet, self).__init__()
        self.nodeMLPList = nn.ModuleList()
        self.num_relation = num_relation
        for i in range(self.num_relation):
            self.nodeMLPList.append(Mlp(in_features=in_features, 
                                        hidden_features=mlp_output_dim, 
                                        out_features=mlp_output_dim))
        self.encoder = Encoder(in_features=mlp_output_dim, 
                               hidden_features=hidden_features, 
                               out_features=out_features, 
                               num_node=num_node, 
                               embeded_dim=embeded_dim,
                               num_relation=num_relation,
                               num_layers=num_layers
                              )
        self.decoder = Decoder(in_features=hidden_features, 
                               hidden_features=hidden_features, 
                               out_features=out_features, 
                               num_node=num_node, 
                               embeded_dim=embeded_dim,
                               num_relation=num_relation,
                               num_layers=num_layers)
        self.linear =  nn.Sequential(
            nn.Linear(hidden_features * num_relation, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        batch_size, seq_time, num_node, in_feat = x.size()
        
        x_relations = []
        for i in range(self.num_relation):
            x_relations.append(self.nodeMLPList[i](x))
        x = torch.stack(x_relations, dim=-1)
        
        out, h_pre = self.encoder(x)
        decode_in = torch.zeros_like(out, device=out.device)
        out, _ = self.decoder(decode_in, h_pre)
        out = out.view(batch_size, seq_time, num_node, -1)
        out = self.linear(out)
        
        return out, x