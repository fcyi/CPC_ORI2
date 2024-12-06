import torch
import torch.nn as nn


class Autoregressor(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(Autoregressor, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 如果为 True，则输入和输出张量以（batch, seq, feature）提供，而不是（seq, batch, feature）
        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        # 重置参数数据指针，以便使用更快的代码路径
        # 可以认为它只是把你所有的weight压缩到一个连续的内存chuck中
        # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据
        # 存放成contiguous chunk(连续的块)。类似我们调用tensor.contiguous
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out
