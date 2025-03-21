import socket
import pickle
import torch
import math
from torch import nn
from dataset import de_vocab, de_preprocess, train_dataset
from emb import EmbeddingWithPosition

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, head):
        super().__init__()
        self.emb_size = emb_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head = head

        self.w_q = nn.Linear(emb_size, head * q_k_size)
        self.w_k = nn.Linear(emb_size, head * q_k_size)
        self.w_v = nn.Linear(emb_size, head * v_size)

        self.kv_cache = {}
        self.kv_cache_type = ''

        # 主机B的 IP 和端口
        self.host_b_address = ('192.168.207.213', 12345)

    def set_kvcache(self, kv_cache_type=''):
        """设置 KV Cache 类型（selfattn 或 crossattn）"""
        self.kv_cache_type = kv_cache_type
        self.kv_cache = {}

    def send_kv_cache_to_host_b(self, kv_cache):
        """发送 KV Cache 到主机 B"""
        try:
            # 序列化数据
            serialized_data = pickle.dumps(kv_cache)
            print("Sending KV Cache to Host B:", kv_cache)  # 调试信息

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)  # 设置超时时间
                s.connect(self.host_b_address)

                # 发送消息头（操作类型 + 数据大小）
                header = b'STORE' + len(serialized_data).to_bytes(4, byteorder='big')
                s.sendall(header)

                # 发送数据
                s.sendall(serialized_data)
                print("KV Cache sent to Host B successfully.")

        except Exception as e:
            print(f"Failed to send KV cache to Host B: {e}")

    def receive_kv_cache_from_host_b(self):
        """从主机 B 接收 KV Cache"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)  # 设置超时时间
                s.connect(self.host_b_address)

                # 发送消息头（操作类型）
                s.sendall(b'RETRI')

                # 接收数据大小（4字节）
                size_data = s.recv(4)
                if not size_data:
                    print("[!] Failed to receive size information")
                    return None
                size = int.from_bytes(size_data, byteorder='big')
                print(f"Expecting {size} bytes of KV Cache from Host B.")

                # 接收数据
                received_data = b''
                while len(received_data) < size:
                    chunk = s.recv(min(4096, size - len(received_data)))
                    if not chunk:
                        raise ConnectionError("Connection closed prematurely.")
                    received_data += chunk

                return pickle.loads(received_data) if received_data else None

        except Exception as e:
            print(f"Failed to receive KV cache from Host B: {e}")
            return None

    def forward(self, x_q, x_k_v, attn_mask):
        """多头注意力计算"""
        if self.kv_cache_type == 'selfattn':  # decoder 自注意力 KV 缓存
            x_q = x_q[:, -1:, :]
            x_k_v = x_k_v[:, -1:, :]

            q = self.w_q(x_q)
            k = self.w_k(x_k_v)
            v = self.w_v(x_k_v)

            if 'Q' in self.kv_cache:
                q = torch.concat((self.kv_cache['Q'], q), dim=1)
            if 'K' in self.kv_cache:
                k = torch.concat((self.kv_cache['K'], k), dim=1)
            if 'V' in self.kv_cache:
                v = torch.concat((self.kv_cache['V'], v), dim=1)

            self.kv_cache.update({'Q': q.detach(), 'K': k.detach(), 'V': v.detach()})

            # 发送一半的 KV Cache 到主机 B
            split_point = max(1, q.size(1) // 2)  # 确保至少有 1 个元素
            half_kv_cache = {
                'Q': q.detach()[:, :split_point, :],
                'K': k.detach()[:, :split_point, :],
                'V': v.detach()[:, :split_point, :]
            }
            self.send_kv_cache_to_host_b(half_kv_cache)

        elif self.kv_cache_type == 'crossattn':  # decoder 交叉注意力 KV 缓存
            x_q = x_q[:, -1:, :]
            q = self.w_q(x_q)

            if 'Q' in self.kv_cache:
                q = torch.concat((self.kv_cache['Q'], q), dim=1)
            self.kv_cache['Q'] = q.detach()

            if 'K' not in self.kv_cache:
                k = self.w_k(x_k_v)
                self.kv_cache['K'] = k.detach()
            else:
                k = self.kv_cache['K']

            if 'V' not in self.kv_cache:
                v = self.w_v(x_k_v)
                self.kv_cache['V'] = v.detach()
            else:
                v = self.kv_cache['V']

            # 从主机 B 加载 KV Cache
            loaded_kv_cache = self.receive_kv_cache_from_host_b()
            if loaded_kv_cache:
                self.kv_cache.update(loaded_kv_cache)

        else:  # 训练模式
            q = self.w_q(x_q)
            k = self.w_k(x_k_v)
            v = self.w_v(x_k_v)

        q = q.view(q.size(0), q.size(1), self.head, self.q_k_size).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.head, self.q_k_size).transpose(1, 2).transpose(2, 3)

        attn = torch.matmul(q, k) / math.sqrt(self.q_k_size)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.head, -1, -1)
        attn = attn.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(attn, dim=-1)

        v = v.view(v.size(0), v.size(1), self.head, self.v_size).transpose(1, 2)
        z = torch.matmul(attn, v)
        z = z.transpose(1, 2)
        return z.reshape(z.size(0), z.size(1), -1)

    def update(self, cache_dict):
        """更新 KV Cache"""
        self.kv_cache.update(cache_dict)


if __name__ == '__main__':
    # 生成一个样本 batch
    emb = EmbeddingWithPosition(len(de_vocab), 128)
    de_tokens, de_ids = de_preprocess(train_dataset[0][0])
    de_ids_tensor = torch.tensor(de_ids, dtype=torch.long)
    emb_result = emb(de_ids_tensor.unsqueeze(0))  # 扩展 batch 维度

    print('emb_result:', emb_result.size())

    # 运行多头注意力
    multihead = MultiHeadAttention(emb_size=128, q_k_size=256, v_size=512, head=8)
    attn_mask = torch.zeros((1, de_ids_tensor.size(0), de_ids_tensor.size(0)))
    multihead_result = multihead(x_q=emb_result, x_k_v=emb_result, attn_mask=attn_mask)

    print('multihead_result:', multihead_result.size())