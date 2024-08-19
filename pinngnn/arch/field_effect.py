import torch
import torch.nn as nn
import numpy as np

# 定义一个可学习的扩散系数
class FieldEffectModel(nn.Module):
    def __init__(self, init_D=0.1):
        super(FieldEffectModel, self).__init__()
        self.D = nn.Parameter(torch.tensor(init_D, dtype=torch.float32))

    def forward(self, field_effect, dt, dx):
        laplacian = (
            torch.roll(field_effect, 1, dims=0) + torch.roll(field_effect, -1, dims=0) +
            torch.roll(field_effect, 1, dims=1) + torch.roll(field_effect, -1, dims=1) -
            4 * field_effect
        ) / (dx * dx)
        field_effect_new = field_effect + self.D * laplacian * dt
        return field_effect_new

# 示例损失函数
def loss_function(predicted_field_effect, actual_field_effect):
    return nn.MSELoss()(predicted_field_effect, actual_field_effect)

# 初始化场效应
def initialize_field_effect(shape, initial_conditions):
    # shape: 场效应的空间尺寸（例如[grid_size_x, grid_size_y]）
    # initial_conditions: 初始条件
    field_effect = np.zeros(shape)
    for condition in initial_conditions:
        x, y, value = condition
        field_effect[x, y] = value
    return field_effect

# 示例训练代码
# 初始化模型
model = FieldEffectModel(init_D=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 设定时间步长和空间步长
dt = 0.01
dx = 1.0

# 生成初始场效应（假设这里有实际数据）
initial_conditions = [(50, 50, 10)]
field_effect = initialize_field_effect((100, 100), initial_conditions)
field_effect = torch.tensor(field_effect, dtype=torch.float32)

# 模拟实际的场效应数据（这里假设一些目标数据）
actual_field_effect = field_effect.clone()  # 在真实场景中，这将是你的观测数据

# 训练模型
for epoch in range(100):  # 100个训练轮次
    optimizer.zero_grad()
    
    # 预测场效应
    predicted_field_effect = model(field_effect, dt, dx)
    
    # 计算损失
    loss = loss_function(predicted_field_effect, actual_field_effect)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 更新场效应
    field_effect = predicted_field_effect.detach().clone()

    print(f'Epoch {epoch}, Loss: {loss.item()}, D: {model.D.item()}')
