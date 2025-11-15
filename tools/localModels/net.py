import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def Rope(x):
    B, T, C = x.shape
    # 计算theta角度
    theta = 10000 ** (-torch.arange(0, C, 2, device=x.device) / C) # （C//2）
    # 生成位置索引
    pos = torch.arange(T, device=x.device).unsqueeze(1) # (T, 1)
    # 计算旋转角度
    angles = pos * theta # (C//2) -> (1, C//2) -> (T, C//2); (T, 1) -> (T, C//2)
    # 计算三角量
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # 拆分数据，两两一组
    x1, x2 = x[:,:,::2], x[:,:,1::2] # (B, T, C//2)
    # 旋转
    rotated_x1 = x1*cos-x2*sin # (B, T, C//2)
    rotated_x2 = x1*sin+x2*cos
    # 还原
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2) # (B, T, C//2, 2) -> (B, T, C)

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.SCALE = 1 # 增益系数

    def forward(self, x, pad_mask=None):
        # x: (batch, seq_len, embed_dim)
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE 需要对每个头的Q,K做，先reshape回(batch*heads, seq_len, head_dim)
        q = q.reshape(B * self.num_heads, T, self.head_dim)
        k = k.reshape(B * self.num_heads, T, self.head_dim)

        q = Rope(q)
        k = Rope(k)

        # reshape回(B, heads, T, head_dim)
        q = q.view(B, self.num_heads, T, self.head_dim)
        k = k.view(B, self.num_heads, T, self.head_dim)

        # scaled dot product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        if pad_mask is not None: # (B, T)
            attn_scores = attn_scores.masked_fill(pad_mask[:, None, None, :] == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, heads, T, T)

        out = torch.matmul(attn_probs, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)  # (B, T, embed_dim)


        out = self.out_proj(out)  # (B, T, embed_dim)
        return out, attn_probs
    
class FFN(nn.Module):
    def __init__(self, embed_dim, drop_p=0.1) -> None:
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.GELU = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(embed_dim * 4, embed_dim)
        self.c_proj.SCALE = 1
        self.drop = nn.Dropout(drop_p)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.GELU(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x
    
class RoPETransformer(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.RoPEAttention = MultiHeadSelfAttentionRoPE(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim)
    
    def forward(self, x, mask=None):
        x1 = self.ln1(x)
        x1, attn = self.RoPEAttention(x1, pad_mask=mask)
        x = x + x1
        x1 = self.ln2(x)
        x = x + self.ffn(x1)
        return x, attn