import torch
import torch.distributed as dist
from torch import nn
import math

# 模拟词向量和预处理步骤
class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(EmbeddingWithPosition, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(5000, emb_size)  # 假设最大序列长度为5000

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        return self.embedding(x) + self.position_embedding(position)

# 模拟数据预处理
de_vocab = ['<pad>', '<bos>', '<eos>', 'ich', 'du', 'er']  # 一个简单的词表
def de_preprocess(text):
    token_to_id = {word: idx for idx, word in enumerate(de_vocab)}
    return [token_to_id[word] for word in text.split()]

train_dataset = [("ich du er",)]  # 一个简单的样本

# 管理分布式 KV Cache
class DistributedKVCache:
    def __init__(self, rank, world_size, split_ratio=0.5):
        self.rank = rank
        self.world_size = world_size
        self.split_ratio = split_ratio
        self.kv_cache = {}  # 本地 KV Cache
        self.remote_kv_cache = {}  # 远端 KV Cache

    def split_kv_cache(self, k, v):
        """根据 split_ratio 划分本地存储部分与远程存储部分"""
        split_idx = int(k.shape[1] * self.split_ratio)  # 计算划分索引
        if self.rank == 0:  # 主机 A 只存储后半部分
            self.remote_kv_cache["K"] = k[:, split_idx:].detach()
            self.remote_kv_cache["V"] = v[:, split_idx:].detach()
            self.kv_cache["K"] = k[:, :split_idx].detach()
            self.kv_cache["V"] = v[:, :split_idx].detach()
        else:  # 主机 B 只存储前半部分
            self.kv_cache["K"] = k[:, split_idx:].detach()
            self.kv_cache["V"] = v[:, split_idx:].detach()
    
    def send_kv_cache(self, dst_rank):
        """主机 A 发送 KV Cache"""
        if self.rank == 0:
            for key, tensor in self.remote_kv_cache.items():
                dist.send(tensor, dst=dst_rank)
                print(f"[主机 A] 已发送 {key}")

    def recv_kv_cache(self, src_rank):
        """主机 B 接收 KV Cache"""
        if self.rank == 1:
            for key in ["K", "V"]:
                tensor = torch.empty_like(self.kv_cache[key])  # 预分配相同大小的 Tensor
                dist.recv(tensor, src=src_rank)
                self.kv_cache[key] = torch.cat((tensor, self.kv_cache[key]), dim=1)
                print(f"[主机 B] 已接收 {key}")

# Multi-Head Attention模块
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, head, rank, world_size):
        super().__init__()
        self.emb_size = emb_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head = head
        self.rank = rank

        self.w_q = nn.Linear(emb_size, head * q_k_size)
        self.w_k = nn.Linear(emb_size, head * q_k_size)
        self.w_v = nn.Linear(emb_size, head * v_size)

        # 分布式 KV Cache
        self.kv_cache = DistributedKVCache(rank, world_size)

    def forward(self, x_q, x_k_v, attn_mask):
        """前向传播时，主机 B 需要从主机 A 拉取 KV Cache"""
        q = self.w_q(x_q)
        k = self.w_k(x_k_v)
        v = self.w_v(x_k_v)

        # 主机 A 进行 KV Cache 分片
        if self.rank == 0:
            self.kv_cache.split_kv_cache(k, v)
            self.kv_cache.send_kv_cache(dst_rank=1)  # 发送给主机 B
        elif self.rank == 1:
            self.kv_cache.recv_kv_cache(src_rank=0)  # 从主机 A 拉取数据
        
        k = self.kv_cache.kv_cache["K"]
        v = self.kv_cache.kv_cache["V"]

        # 计算注意力
        q = q.view(q.size(0), q.size(1), self.head, self.q_k_size).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.head, self.q_k_size).transpose(1, 2).transpose(2, 3)
        attn = torch.matmul(q, k) / math.sqrt(self.q_k_size)
        attn = attn.masked_fill(attn_mask.unsqueeze(1).expand(-1, self.head, -1, -1), -1e9)
        attn = torch.softmax(attn, dim=-1)
        v = v.view(v.size(0), v.size(1), self.head, self.v_size).transpose(1, 2)
        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).reshape(z.size(0), z.size(1), -1)
        return z

# 初始化分布式环境
def init_distributed(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 设置当前进程的 GPU（如果在多 GPU 环境中运行）

if __name__ == '__main__':
    rank = 1  # 主机A
    world_size = 2  # 假设有两台机器

    # 初始化分布式环境
    init_distributed(rank, world_size)

    # 准备1个batch
    emb = EmbeddingWithPosition(len(de_vocab), 128).cuda(rank)
    de_tokens, de_ids = de_preprocess(train_dataset[0][0])  # 取 de 句子转词 ID 序列
    de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).cuda(rank)  # 将数据移动到 GPU
    emb_result = emb(de_ids_tensor.unsqueeze(0))  # 转 batch 再输入模型
    print('emb_result:', emb_result.size())

    # 多头注意力
    multihead = MultiHeadAttention(emb_size=128, q_k_size=256, v_size=512, head=8, rank=rank, world_size=world_size)
    attn_mask = torch.zeros((1, de_ids_tensor.size()[0], de_ids_tensor.size()[0])).cuda(rank)  # batch 中每个样本对应 1 个注意力矩阵
    multihead_result = multihead(x_q=emb_result, x_k_v=emb_result, attn_mask=attn_mask)
    print('multihead_result:', multihead_result.size())

    # 结束分布式环境
    dist.barrier()  # 等待其他进程
    dist.destroy_process_group()  # 关闭分布式进程
