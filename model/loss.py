import torch
import torch.nn.functional as F

class MultiDistanceLoss(torch.nn.Module):
    def __init__(self, margin, relation):
        super(MultiDistanceLoss, self).__init__()
        self.margin = margin
        self.relation = relation
        
    def forward(self, x):
        total_loss = 0.0
        
        x_relations = []
        
        for r in range(self.relation):
            x_relations.append(F.normalize(x[...,r], dim=-1))
        
        total_loss = 0
        
        if self.relation >= 3:
            for r in range(self.relation):
                for j in range(self.relation):
                    i = (r+1)%self.relation
                    if j != i and j != r:
#                         print(r, i, j)
                        anchor = x_relations[r]
                        positive = x_relations[i]
                        negative = x_relations[j]
                        dis_positive = torch.nn.functional.pairwise_distance(anchor, positive)
                        dis_negative= torch.nn.functional.pairwise_distance(anchor, negative)
                        total_loss += torch.relu(dis_positive - dis_negative + self.margin)
            return total_loss.mean()
        elif self.relation == 2:
            dis_negative = torch.nn.functional.pairwise_distance(x_relations[0], x_relations[1])
            total_loss += torch.relu(dis_negative + self.margin)
            return total_loss.mean()
        else:
            return 0
