'''
输入emb后的词序列,根据Q,K,V方法计算词与词之间的相关性,为每个词生成信息提取后的emb(与输入词1:1映射)
'''
import socket
import pickle
import torch
import math
import time
from torch import nn
from dataset import de_vocab, de_preprocess, train_dataset
from emb import EmbeddingWithPosition
import pandas as pd

class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size,q_k_size,v_size,head):
        super().__init__()
        self.emb_size=emb_size
        self.q_k_size=q_k_size
        self.v_size=v_size
        self.head=head

        self.w_q=nn.Linear(emb_size,head*q_k_size) # 多头
        self.w_k=nn.Linear(emb_size,head*q_k_size)
        self.w_v=nn.Linear(emb_size,head*v_size)
        
        # kvcache推理优化
        self.kv_cache={}
        self.kv_cache1={} 
        self.local_kv_cache=None
        self.kv_cache_type=''
        self.kv_cache_type1=''
        self.host_b_address = ('192.168.207.213', 12345)
        self.log_data = []
    def set_kvcache(self,kv_cache_type=''):
        self.kv_cache_type=kv_cache_type
        self.kv_cache={}

    def set_kvcache1(self,kv_cache_type1=''):
        self.kv_cache_type1=kv_cache_type1
        self.kv_cache1={}   

    def save_log_to_excel(self, filename='kv_cache_log.xlsx'):
        df = pd.DataFrame(self.log_data, columns=['Type', 'Size (bytes)', 'Time (sec)'])
        df.to_excel(filename, index=False)
        print(f"[LOG] KV Cache log saved to {filename}")

    def send_kv_cache_to_host_b(self, kv_cache):
        """发送 KV Cache 到主机 B"""
        try:
            serialized_data = pickle.dumps(kv_cache)
            start_time = time.time()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect(self.host_b_address)
                header = b'STORE' + len(serialized_data).to_bytes(4, byteorder='big')
                s.sendall(header)
                s.sendall(serialized_data)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[SEND] KV Cache Sent | Size: {len(serialized_data)} bytes | Time: {elapsed_time:.6f} sec")
            self.log_data.append(['SEND', len(serialized_data), elapsed_time])
        except Exception as e:
            print(f"[SEND] Failed to send KV cache: {e}")

    def receive_kv_cache_from_host_b(self):
        """从主机 B 接收 KV Cache"""
        try:
            start_time = time.time()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect(self.host_b_address)
                s.sendall(b'RETRI')
                size_data = s.recv(4)
                if not size_data:
                    print("[RECEIVE] Failed to receive size information")
                    return None
                size = int.from_bytes(size_data, byteorder='big')
                #print(f"[RECEIVE] KV Cache Expected Size: {size} bytes")
                received_data = b''
                while len(received_data) < size:
                    chunk = s.recv(min(4096, size - len(received_data)))
                    if not chunk:
                        raise ConnectionError("[RECEIVE] Connection closed prematurely.")
                    received_data += chunk
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"{elapsed_time:.6f}")
                self.log_data.append(['RECEIVE', len(received_data), elapsed_time])
                return pickle.loads(received_data) if received_data else None
        except Exception as e:
            print(f"[RECEIVE] Failed to receive KV cache: {e}")
            return None
   
   
    def forward(self,x_q,x_k_v,attn_mask):  # x_q: (batch_size,seq_len,emb_size), x_k_v:  (batch_size,seq_len',emb_size)
        # kvcache推理加速,只有decoder推理阶段使用
    
        if self.kv_cache_type=='selfattn': # decoder的自注意力cache
            x_q=x_q[:,-1:,:] # (batch_size,seq_len=1,emb_size)
            x_k_v=x_k_v[:,-1:,:] # (batch_size,seq_len'=1,emb_size)
            
            q=self.w_q(x_q) # q: (batch_size,seq_len=1,head*q_k_size)
            k=self.w_k(x_k_v) # k: (batch_size,seq_len=1,head*q_k_size)
            v=self.w_v(x_k_v) # v: (batch_size,seq_len=1,head*v_size)
            if 'Q' in self.kv_cache:
                q=torch.concat((self.kv_cache['Q'],q),dim=1) # 追加到前一次推理的Q末尾
            if 'K' in self.kv_cache:
                k=torch.concat((self.kv_cache['K'],k),dim=1) # 追加到前一次推理的K末尾
            if 'V' in self.kv_cache:
                v=torch.concat((self.kv_cache['V'],v),dim=1) # 追加到前一次推理的K末尾
            self.update({'Q':q.detach(),'K':k.detach(),'V':v.detach()}) # 更新缓存
           # self.send_kv_cache_to_host_b(self.kv_cache)

        elif self.kv_cache_type=='crossattn': # decoder的交叉注意力cache
            x_q=x_q[:,-1:,:] # (batch_size,seq_len=1,emb_size)
            q=self.w_q(x_q) # q: (batch_size,seq_len,head*q_k_size)

            if self.local_kv_cache is None:
                #print(2)
                loaded_kv_cache = self.receive_kv_cache_from_host_b()
                self.local_kv_cache = loaded_kv_cache
                
            if 'Q' in self.kv_cache:
              #  self.kv_cache['Q']=torch.concat((self.kv_cache['Q'], loaded_kv_cache['Q']),dim=1) # 追加到前一次推理的K末尾
                q=torch.concat((self.kv_cache['Q'],q),dim=1) # 追加到前一次推理的Q末尾           

            if self.local_kv_cache is not None and 'K' in self.local_kv_cache:
               # print(1)
                k=self.local_kv_cache['K']
            elif 'K' in self.kv_cache:
                k=self.kv_cache['K']
            else:
                k=self.w_k(x_k_v) # k: (batch_size,seq_len,head*q_k_size)

            if self.local_kv_cache is not None and 'V' in self.local_kv_cache:
              #  print(1)
                v=self.local_kv_cache['V']
            elif 'V' in self.kv_cache:
                v=self.kv_cache['V']  
            else:
                v=self.w_v(x_k_v) # v: (batch_size,seq_len,head*v_size)

            self.kv_cache.update({'Q':q.detach(),'K':k.detach(),'V':v.detach()}) # 更新缓存
            
        else: # 训练模式
            q=self.w_q(x_q) # q: (batch_size,seq_len,head*q_k_size)
            k=self.w_k(x_k_v) # k: (batch_size,seq_len,head*q_k_size)
            v=self.w_v(x_k_v) # v: (batch_size,seq_len,head*v_size)
            if self.kv_cache_type1=='selfattn':
                self.update1({'Q':q.detach(),'K':k.detach(),'V':v.detach()}) # 更新缓存
                self.send_kv_cache_to_host_b(self.kv_cache1)

        # 多头兼容
        q=q.view(q.size()[0],q.size()[1],self.head,self.q_k_size).transpose(1,2) # q: (batch_size,head,seq_len,q_k_size)
        k=k.view(k.size()[0],k.size()[1],self.head,self.q_k_size).transpose(1,2).transpose(2,3) # k:(batch_size,head,q_k_size,seq_len)
       
        # 注意力矩阵
        attn=torch.matmul(q,k)/math.sqrt(self.q_k_size) # (batch_size,head,seq_len,seq_len) row是q,col是k
        
        # 注意力分值处理
        # attn_mask: (batch_size,seq_len,seq_len)
        attn_mask=attn_mask.unsqueeze(1).expand(-1,self.head,-1,-1) # attn_mask: (batch_size,head,seq_len,seq_len)
        attn=attn.masked_fill(attn_mask,-1e9)
        attn=torch.softmax(attn,dim=-1) # scores: (batch_size,head,seq_len,seq_len)
        
        # 注意力与V相乘
        v=v.view(v.size()[0],v.size()[1],self.head,self.v_size).transpose(1,2) # v: (batch_size,head,seq_len,v_size)
        z=torch.matmul(attn,v) # z: (batch_size,head,seq_len,v_size)
        z=z.transpose(1,2) # z: (batch_size,seq_len,head,v_size)
        return z.reshape(z.size()[0],z.size()[1],-1) # z: (batch_size,seq_len,head*v_size)
    
    def update(self, cache_dict):#
        """更新 KV Cache"""
        self.kv_cache.update(cache_dict)
    def update1(self, cache_dict):#
        """更新 KV Cache"""
        self.kv_cache1.update(cache_dict)


    
if __name__=='__main__':
    # 准备1个batch
    emb=EmbeddingWithPosition(len(de_vocab),128)
    de_tokens,de_ids=de_preprocess(train_dataset[0][0]) # 取de句子转词ID序列
    de_ids_tensor=torch.tensor(de_ids,dtype=torch.long)
    emb_result=emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    print('emb_result:', emb_result.size())

    # 多头注意力
    multihead=MultiHeadAttention(emb_size=128,q_k_size=256, v_size=512, head=8)
    multihead.save_log_to_excel('kv_cache_log.xlsx')
    attn_mask=torch.zeros((1,de_ids_tensor.size()[0],de_ids_tensor.size()[0])) # batch中每个样本对应1个注意力矩阵
    multihead_result=multihead(x_q=emb_result,x_k_v=emb_result,attn_mask=attn_mask)
    print('multihead_result:', multihead_result.size())
    