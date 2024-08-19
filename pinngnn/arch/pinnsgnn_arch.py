import torch
import torch.nn as nn
from .pinns_layers import PINNBlock, OutputBlock

class PINNsGNN(nn.Module):
    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate):
        super(PINNsGNN, self).__init__()
        self.num_input_features = 3
        self.output_dim = blocks[-1][0]
        self.n_vertex = n_vertex
        
        modules = []
        for l in range(len(blocks) - 3):
            in_channels = self.num_input_features if l == 0 else blocks[l][-1]
            modules.append(PINNBlock(
                Kt, Ks, n_vertex, in_channels, blocks[l+1], act_func, graph_conv_type, gso, bias, droprate))
        self.st_blocks = nn.Sequential(*modules)
        
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        self.output = OutputBlock(
            Ko, blocks[-3][-1], blocks[-2], self.output_dim, n_vertex, act_func, bias, droprate)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        batch_size, _, num_nodes, _ = history_data.shape
        x = history_data.permute(0, 3, 1, 2).contiguous()
        x = self.st_blocks(x)
        x = self.output(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return {"prediction": x}
    
    def calculate_physics_loss(self, pred, true, occupy, speed):
        # physics constraint
        estimated_flow = (occupy / 100) * speed
        physics_loss = nn.MSELoss()(estimated_flow, true) + nn.MSELoss()(pred, true)
        return physics_loss