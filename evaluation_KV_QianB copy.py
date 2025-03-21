# host_b.py
import torch
import socket
import pickle
import threading
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
from dataset import BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX,en_vocab,de_vocab

import time
class MigrationServer:
    def __init__(self, model_path):
        self.model = Transformer(enc_vocab_size=len(de_vocab),dec_vocab_size=len(en_vocab),emb_size=512,q_k_size=64,v_size=64,f_size=2048,head=8,nblocks=6,dropout=0.1,seq_max_len=SEQ_MAX_LEN).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.current_jobs = {}
        self.lock = threading.Lock()
    
    def start(self, host='0.0.0.0', port=12345):
        """启动迁移服务"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            print(f"[HostB] 迁移服务已启动在 {host}:{port}")
            
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_connection, args=(conn, addr)).start()
    
    def handle_connection(self, conn, addr):
        """处理迁移请求"""
        try:
            # 读取协议头
            magic = conn.recv(4)  # 读取4字节的协议头
            if magic != b'MIGR':
               print(f"[HostB] 无效协议头: {magic}")
               return
            # 读取数据长度（后4字节）
            data_length_bytes = conn.recv(4)
            if len(data_length_bytes) != 4:
                print(f"[HostB] 数据长度字段不完整")
                return
            

            data_length = int.from_bytes(data_length_bytes, 'big')
        
            # 接收完整数据
            received_data = b''
            while len(received_data) < data_length:
                chunk = conn.recv(data_length - len(received_data))
                if not chunk:
                    break
                received_data += chunk       
            # 确认数据完整
            if len(received_data) != data_length:
                print(f"[HostB] 数据不完整，预期长度 {data_length}，实际 {len(received_data)}")
                return     
            # 反序列化数据
            data = pickle.loads(received_data)
            # 发送确认
            conn.sendall(b'ACK')

            # 处理迁移数据
            with self.lock:
                job_id = time.time_ns()
                self.current_jobs[job_id] = data
                print(f"[HostB] 收到来自 {addr} 的迁移任务 {job_id}")
                
                # 启动继续生成
                result = self.continue_translation(data)
                
                del self.current_jobs[job_id]
                print(f"[HostB] 任务 {job_id} 完成: {result}")
                return result
                
        except Exception as e:
            print(f"[HostB] 处理 {addr} 时发生错误: {str(e)}")
        finally:
            conn.close()
    
    def continue_translation(self, data):
        """继续翻译执行"""
        # 加载编码结果
        encoder_z = data['encoder_z'].to(DEVICE)
        en_token_ids = data['current_tokens']
        enc_x = data['enc_x'].to(DEVICE)
        # 初始化解码器
        self.model.decoder.open_kvcache()
        print(f'[HostB]enc_x:{enc_x.shape}, encoder_z, 形状: {encoder_z.shape}')
                
        
        # 加载各层缓存
        for layer_idx, layer in enumerate(self.model.decoder.decoder_blocks):
            layer_key = f'layer_{layer_idx}'
            if layer_key in data['kv_cache']:
                layer.first_multihead_attn.kv_cache['K'] = data['kv_cache'][layer_key]['self_attn']['K'].to(DEVICE)
                layer.first_multihead_attn.kv_cache['V'] = data['kv_cache'][layer_key]['self_attn']['V'].to(DEVICE)
                
                layer.second_multihead_attn.kv_cache['K'] = data['kv_cache'][layer_key]['cross_attn']['K'].to(DEVICE)
                layer.second_multihead_attn.kv_cache['V'] = data['kv_cache'][layer_key]['cross_attn']['V'].to(DEVICE)           
               
                cross_attn_k = data['kv_cache'][layer_key]['cross_attn']['K']
                self_attn_k =data['kv_cache'][layer_key]['self_attn']['K']
                print(f"[HostB] 层 {layer_idx} 原始交叉注意力缓存形状: {cross_attn_k.shape}")
                print(f"[HostB] 层 {layer_idx} 原始自注意力缓存形状: {self_attn_k.shape}")
                #print(f"[HostB] 层 {layer_idx} 填充后交叉注意力缓存形状: {layer.second_multihead_attn.kv_cache['K'].shape}")
        
        
        #张量输出相同
        try:
            # 继续生成循环
            while len(en_token_ids)<SEQ_MAX_LEN:
                 dec_x_batch=torch.tensor([en_token_ids],dtype=torch.long).to(DEVICE)  # 准备decoder输入
                 decoder_z=self.model.decode(dec_x_batch,encoder_z,enc_x)   # decoder解碼
                 next_token_probs=decoder_z[0,dec_x_batch.size(-1)-1,:]    # 序列下一个词的概率
                 next_token_id=torch.argmax(next_token_probs)    # 下一个词ID
                 en_token_ids.append(next_token_id)

                 if next_token_id==EOS_IDX:  # 结束符
                     break
            self.model.decoder.close_kvcache() # 清理KV Cache
    
             # 生成翻译结果
            en_token_ids=[id for id in en_token_ids if id not in [BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX]] # 忽略特殊字符
            en_tokens=en_vocab.lookup_tokens(en_token_ids)    # 词id序列转token序列
            return ' '.join(en_tokens)

        finally:
            self.model.decoder.close_kvcache()

if __name__ == '__main__':
    server = MigrationServer('checkpoints/model.pth')
    server.start()