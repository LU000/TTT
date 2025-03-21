'''
输入emb后的词序列,根据Q,K,V方法计算词与词之间的相关性,为每个词生成信息提取后的emb(与输入词1:1映射)
'''
import socket
import pickle
import torch
import math
import time
import os
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
        self.local_kv_cache = None
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
        """将发送和接收的KV Cache数据按指定格式保存到Excel"""
        if not self.log_data:
          #  print("[LOG] No new KV Cache data to save.")
            return
    
    # 处理成对的发送和接收数据
        paired_rows = []
        i = 0
        while i < len(self.log_data):
        # 确保至少有两个条目可配对
            if i + 1 >= len(self.log_data):
                 break
            
            send_entry = self.log_data[i]
            receive_entry = self.log_data[i+1]
        
        # 验证是否为有效的发送-接收对
            if send_entry[0] == 'SEND' and receive_entry[0] == 'RECEIVE':
                paired_rows.append([
                    send_entry[1], send_entry[2],  # 发送大小和时间
                    receive_entry[1], receive_entry[2]  # 接收大小和时间
                ])
                i += 2  # 跳过已处理的条目
            else:
                i += 1  # 不匹配则继续检查下一个
    
        if not paired_rows:
            print("[LOG] No valid SEND/RECEIVE pairs found.")
            return
    
    # 创建符合要求的DataFrame
        columns = ['Send Size (bytes)', 'Send Time (sec)', 
                   'Receive Size (bytes)', 'Receive Time (sec)']
        new_df = pd.DataFrame(paired_rows, columns=columns)
    
    # 合并历史数据（如果存在）
        if os.path.exists(filename):
            try:
                old_df = pd.read_excel(filename)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
            except:
                combined_df = new_df
        else:
            combined_df = new_df
    
    # 写入Excel文件
        combined_df.to_excel(filename, index=False)
        print(f"[LOG] Saved KV Cache log with {len(paired_rows)} pairs to {filename}")
        self.log_data.clear()  # 清空已处理数据

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
            #print(f"[SEND] KV Cache Sent | Size: {len(serialized_data)} bytes | Time: {elapsed_time:.6f} sec")
            print(f"{len(serialized_data)}     {elapsed_time:.6f}")
            #print(f"{elapsed_time:.6f}")
            self.log_data.append(['SEND', len(serialized_data), elapsed_time])
        except Exception as e:
            print(f"[SEND] Failed to send KV cache: {e}")

    def receive_kv_cache_from_host_b(self, Q):
        """从主机 B 接收 KV Cache"""
        try:
            start_time = time.time()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect(self.host_b_address)

                # 序列化 Q 并计算大小
                q_data = pickle.dumps(Q)
                q_size = len(q_data) 
                #print(f"{q_size}")
               # 发送操作头: RETRI + 总数据长度 (Q大小4字节 + Q数据)
                total_size = 4 + q_size  # 4字节Q大小 + Q数据长度
                header = b'RETRI' + total_size.to_bytes(4, byteorder='big')
                s.sendall(header)
                # 发送 Q 大小和 Q 数据
                s.sendall(q_size.to_bytes(4, byteorder='big'))  # 注意此处为大端
                s.sendall(q_data)
                # 先发送 'RETRI' 命令（5字节），再发送 Q 的大小（4字节）和 Q 数据
                #s.sendall(retri_command + q_size + serialized_Q)

                #s.sendall(retri_command + q_size + serialized_Q)  # 发送 RETRI + Q

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
                #print(f"[RECEIVE] KV Cache Received | Size: {len(received_data)} bytes | Time: {elapsed_time:.6f} sec")
                #print(f"{len(received_data)}      {elapsed_time:.6f} ")
                print(f"{elapsed_time:.6f} ")
                received_data = pickle.loads(received_data) if received_data else None
                tensor_data = torch.tensor(received_data)
                #print(f'{tensor_data.size()}')
                self.log_data.append(['RECEIVE', len(received_data), elapsed_time])
                return tensor_data
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

           # if self.local_kv_cache is None:
           #     loaded_kv_cache = self.receive_kv_cache_from_host_b()
           #     self.local_kv_cache = loaded_kv_cache
      
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
            if self.kv_cache_type1=='selfattn1':
                self.update1({'Q':q.detach(),'K':k.detach(),'V':v.detach()}) # 更新缓存
                self.send_kv_cache_to_host_b(self.kv_cache1)

        if self.kv_cache_type =='crossattn':      
            #print(1)
            recieve_z = self.receive_kv_cache_from_host_b(q)
            return recieve_z

          #  print(f"🔹 [received_data] Shape: {received_data.shape()}\n")
            #print(f'{self.head} {self.q_k_size} {self.v_size}')
           # print(f"[DEBUG] Q原始形状: {q.shape} | 总元素数: {q.numel()}")
           # print(f"[DEBUG] K原始形状: {k.shape} | 总元素数: {k.numel()}")
           # print(f"[DEBUG] V原始形状: {v.shape} | 总元素数: {v.numel()}")
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
        #print(z.size())
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
    
    attn_mask=torch.zeros((1,de_ids_tensor.size()[0],de_ids_tensor.size()[0])) # batch中每个样本对应1个注意力矩阵
    multihead_result=multihead(x_q=emb_result,x_k_v=emb_result,attn_mask=attn_mask)
    print('multihead_result:', multihead_result.size())
     
    
    