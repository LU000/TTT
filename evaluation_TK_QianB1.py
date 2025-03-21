# host_b.py
import torch
import socket
import pickle
import threading
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
from dataset import BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX,en_vocab,de_vocab

import time
# host_b.py (修改后的接收端)
class HybridMigrationServer:
    def __init__(self, model_path):
        self.model = Transformer(enc_vocab_size=len(de_vocab),dec_vocab_size=len(en_vocab),emb_size=512,q_k_size=64,v_size=64,f_size=2048,head=8,nblocks=6,dropout=0.1,seq_max_len=SEQ_MAX_LEN).to(DEVICE) 
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.current_jobs = {}
        self.lock = threading.Lock()
        self._init_kv_cache()  # 初始化空缓存
    
    def _init_kv_cache(self):
        """初始化所有层的KV缓存结构"""
        for layer in self.model.decoder.decoder_blocks:
            layer.first_multihead_attn.kv_cache = {'K': torch.tensor([]), 'V': torch.tensor([])}
            layer.second_multihead_attn.kv_cache = {'K': torch.tensor([]), 'V': torch.tensor([])}
    
    def start(self, host='0.0.0.0', port=12345):
        """启动迁移服务"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            print(f"[HostB] 混合迁移服务已启动在 {host}:{port}")
            
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_connection, args=(conn, addr)).start()
    
    def _receive_all(self, conn, length):
        """可靠接收数据"""
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                break
            data += packet
        return pickle.loads(data)
    
    def handle_connection(self, conn, addr):
        """处理迁移请求"""
        try:
            start_time=time.time()          
            # 协议头处理
            magic = conn.recv(4)
            if magic != b'MIGR':
                print(f"[HostB] 无效协议头: {magic}")
                return
            data_length = int.from_bytes(conn.recv(4), 'big')
            
            # 接收完整数据
            data = self._receive_all(conn, data_length)
            conn.sendall(b'ACK')  # 确认接收
            
          
            # 处理数据
            with self.lock:
                job_id = time.time_ns()
                self.current_jobs[job_id] = data
                #print(data)

                result = self.hybrid_continue(data)
                del self.current_jobs[job_id]
                print(f"[HostB] 任务 {job_id} 完成: {result}")
                endtime=time.time()
                print(f"[HostB] 完成时间: {endtime - start_time:.6f} 秒")
              
                return result
                
        except Exception as e:
            print(f"[HostB] 处理 {addr} 时发生错误: {str(e)}")
        finally:
            conn.close()  


    def hybrid_continue(self, data):
        enc_x = data['enc_x'].to(DEVICE)
        token_part = data['token_part']
        kv_data = data['kv_part']
        #recv_time = time.time()
        #start_time = data['start_time']
        #print(f"[HostB] 数据迁移时间: {recv_time - start_time:.6f} 秒")
  
        # 阶段1：重新计算Token部分的KV
        self.model.decoder.open_kvcache()
        encoder_z = self.model.encode(enc_x)
        #print(f'{len(token_part)}')
        
        # 重建前部分KV
        #for i in range(1, len(token_part)+1):
        #    dec_input = torch.tensor([token_part[:i]], dtype=torch.long).to(DEVICE)
        #    _ = self.model.decode(dec_input, encoder_z, enc_x)
        dec_input = torch.tensor([token_part], dtype=torch.long).to(DEVICE)
        _ = self.model.decode(dec_input, encoder_z, enc_x)
       
        # 阶段2：拼接迁移的KV缓存
        for layer_idx in range(len(self.model.decoder.decoder_blocks)):
            layer = self.model.decoder.decoder_blocks[layer_idx]
            migrated_cache = kv_data['cache'][f'layer_{layer_idx}']
            
            # 自注意力拼接
            self_attn = layer.first_multihead_attn
            self._concat_kv(
                self_attn.kv_cache, 
                migrated_cache['self_attn'],
                kv_data['start_idx']
            )
            
            # 交叉注意力拼接
            cross_attn = layer.second_multihead_attn
            self._concat_kv(
                cross_attn.kv_cache,
                migrated_cache['cross_attn'],
                kv_data['start_idx']
            )
        
        # 阶段3：继续生成
        current_seq = data['current_tokens'] # 占位符
        return self._generate_remaining(enc_x,encoder_z, current_seq)
     
    def _receive_all(self, conn, length):
        """可靠接收数据"""
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                break
            data += packet
        return pickle.loads(data)
     
    def _concat_kv(self, target_cache, migrated_cache, start_idx):
        """KV缓存拼接核心逻辑"""
        for key in ['K', 'V']:
            original = target_cache[key]
            migrated = migrated_cache[key].to(DEVICE)
            
            # 维度对齐检查
            if original.dim() == 3 and migrated.dim() == 3:
                if original.size(1) < start_idx:
                    padding = torch.zeros(
                        original.size(0),
                        start_idx - original.size(1),
                        original.size(2)
                    ).to(DEVICE)
                    original = torch.cat([original, padding], dim=1)
                
                # 拼接新缓存
                target_cache[key] = torch.cat([
                    original[:, :start_idx, :],
                    migrated
                ], dim=1)

    def _generate_remaining(self, enc_x,encoder_z, en_token_ids):
        """生成剩余Token"""
        try:
            #encoder_z = self.model.encode(enc_x)
            print(en_token_ids)
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
    starttime=time.time()
    server = HybridMigrationServer('checkpoints/model.pth')
    server.start() 
    endtime=time.time()
    print(f"完成时间={starttime-endtime}s")