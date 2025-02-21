import torch
from torch import Tensor, nn

def gram_schmidt(input: Tensor) -> Tensor:
    """
    对输入张量中的向量组进行Gram-Schmidt正交化。

    :param input: 输入张量，形状为 (N, C, H, W) 或其他适用形状
    :return: 正交化后的张量，形状与输入相同
    """
    def projection(u: Tensor, v: Tensor) -> Tensor:
        return (torch.dot(u.view(-1), v.view(-1)) / torch.dot(u.view(-1), u.view(-1))) * u

    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x / x.norm(p=2)
        output.append(x)
    return torch.stack(output)


def initialize_orthogonal_filters(c: int, h: int, w: int) -> Tensor:
    """
    初始化正交滤波器。

    :param c: 通道数
    :param h: 高度
    :param w: 宽度
    :return: 正交化后的滤波器张量，形状为 (c, 1, h, w)
    """
    if h * w < c:
        n = c // (h * w)
        gram = []
        for _ in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))


class GramSchmidtTransform(nn.Module):
    """
    Gram-Schmidt 变换模块，用于生成正交滤波器。
    使用单例模式确保相同参数只初始化一次，节省资源。
    """
    instance = {}  # 单例存储字典

    def __init__(self, c: int, h: int):
        """
        初始化 GramSchmidtTransform 模块。

        :param c: 通道数
        :param h: 高度（假设 H=W）
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
            self.register_buffer("constant_filter", rand_ortho_filters.to(self.device).detach())

    @staticmethod
    def build(c: int, h: int) -> 'GramSchmidtTransform':
        """
        构建 GramSchmidtTransform 模块的单例实例。

        :param c: 通道数
        :param h: 高度
        :return: GramSchmidtTransform 实例
        """
        if (c, h) not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播，将输入与正交滤波器相乘并在空间维度上求和。

        :param x: 输入张量，形状为 (B, C, H, W) 或 (B, C, H)（对于缺少宽度的三维输入）
        :return: 输出张量，形状为 (B, C, 1, 1)
        """
        # 如果输入是三维的，添加一个维度
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)  # 将三维输入变成四维：(B, C, H) -> (B, C, H, 1)

        B, C, H, W = x.shape  # 现在可以正确解包
        _, H_filter, W_filter = self.constant_filter.shape  # 获取正交滤波器的高宽

        # 如果输入的 H 和 W 与正交滤波器的 H 和 W 不匹配，使用自适应池化
        if H != H_filter or W != W_filter:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H_filter, W_filter))

        # 将输入和正交滤波器相乘并进行求和
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)


class Orthogonal_Channel_Attention(nn.Module):
    """
    正交通道注意力模块，通过正交化增强通道间的关系。
    """
    def __init__(self, channels: int, height: int):
        """
        初始化 Orthogonal_Channel_Attention 模块。

        :param channels: 输入张量的通道数 (C)
        :param height: Gram-Schmidt 变换所需的高度 (假设 H=W)
        """
        super(Orthogonal_Channel_Attention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channels = channels
        self.height = height

        # 初始化 Gram-Schmidt 变换模块
        self.F_C_A = GramSchmidtTransform.build(channels, height)

        # 通道注意力映射（类似 SE Block 结构）
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播。

        :param x: 输入张量 (B, C, H, W)
        :return: 输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 如果输入的 H 和 W 与初始化不匹配，则自适应池化
        if H != self.height or W != self.height:
            x = nn.functional.adaptive_avg_pool2d(x, (self.height, self.height))

        # Gram-Schmidt 变换
        transformed = self.F_C_A(x)  # (B, C, 1, 1)

        # 去除空间维度，进入通道注意力网络
        compressed = transformed.view(B, C)  # (B, C)

        # 通道注意力生成
        excitation = self.channel_attention(compressed).view(B, C, 1, 1)  # (B, C, 1, 1)

        # 加权原始输入特征
        output = x * excitation  # (B, C, H, W)

        return output
