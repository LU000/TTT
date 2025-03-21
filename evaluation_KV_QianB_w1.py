import torch
import socket
import pickle
import threading
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
from dataset import BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, en_vocab, de_vocab
import time

class MigrationServer:
    def __init__(self, model_path):
        self.model = Transformer(enc_vocab_size=len(de_vocab), dec_vocab_size=len(en_vocab),
                               emb_size=512, q_k_size=64, v_size=64, f_size=2048,
                               head=8, nblocks=6, dropout=0.1, seq_max_len=SEQ_MAX_LEN).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.current_jobs = {}
        self.job_data = {'encoder_z': None, 'enc_x': None, 'kv_cache': {}}
        self.lock = threading.Lock()

    def start(self, host='0.0.0.0', port=12345):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            print(f"[HostB] 迁移服务启动于 {host}:{port}")

            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_connection, args=(conn, addr)).start()

    def handle_connection(self, conn, addr):
        try:
            start_time=time.time()
            while True:
                header = conn.recv(4)
                if not header: break

                if header == b'DONE':
                    print("[HostB] 收到完成信号")
                    conn.sendall(b'ACK')
                    break

                elif header == b'MIGR':
                    print("MIGR")
                     # 读取数据长度（后4字节）
                    data_length_bytes = conn.recv(4)
                    if len(data_length_bytes) != 4:
                        print(f"[HostB] 数据长度字段不完整")
                        return
             

                    data_length = int.from_bytes(data_length_bytes, 'big')
            
                    # 接收完整数据
                    received_data = b''

                    while len(received_data) < data_length:
                        #print(f"{len(received_data)},{data_length}")
                        chunk = conn.recv(data_length - len(received_data))
                        if not chunk:
                             break
                        received_data += chunk   

                    migration_data = pickle.loads(received_data)
                    
                    print("发送ACK")
                    conn.sendall(b'ACK')

                    with self.lock:
                        # 合并KV缓存
                      
                        self.merge_kv_cache(self.job_data['kv_cache'], migration_data['kv_cache'])
                        
                        if self.job_data['encoder_z'] is None:
                            self.job_data['encoder_z'] = migration_data['encoder_z'].to(DEVICE)
                            self.job_data['enc_x'] = migration_data['enc_x'].to(DEVICE)
                        if 'current_tokens' in migration_data:
                            self.job_data['current_tokens'] = migration_data['current_tokens']
                            self.job_data['input_len'] = migration_data['input_len']
    
            # 继续解码
            with self.lock:
                result = self.continue_translation(self.job_data)
                print(f"[HostB] 最终翻译结果: {result}")
            
            endtime=time.time()
            print(f"[HostB] 完成时间: {endtime - start_time:.6f} 秒")

        except Exception as e:
            print(f"[HostB] 处理错误: {e}")
        finally:
            conn.close()

    def merge_kv_cache(self, target, new_kv):
      # 在合并操作之前，确保数据都在相同的设备上
      
      for layer_key, attn_data in new_kv.items():
          if layer_key not in target:
              target[layer_key] = {'self_attn': {'K': None, 'V': None}, 
                             'cross_attn': {'K': None, 'V': None}}

          for attn_type in ['self_attn', 'cross_attn']:
              for k_v in ['K', 'V']:
                  if target[layer_key][attn_type][k_v] is None:
                      # 将新的 KV 缓存移动到相同的设备
                      target[layer_key][attn_type][k_v] = new_kv[layer_key][attn_type][k_v].to(DEVICE)
                  else:
                      #print(f"接收到的 KV 缓存形状11: {target[layer_key][attn_type][k_v].shape},{new_kv[layer_key][attn_type][k_v].shape}")
                     # print(f"接收到的 KV 缓存形状11: {target[layer_key][attn_type][k_v].shape},{new_kv[layer_key][attn_type][k_v].shape}")
 
                      # 确保目标和新数据在相同的设备上
                      target[layer_key][attn_type][k_v] = torch.cat(
                          [target[layer_key][attn_type][k_v].to(DEVICE), new_kv[layer_key][attn_type][k_v].to(DEVICE)], dim=1)
                      
                      #print(f"接收到的 KV 缓存形状2: {target[layer_key][attn_type][k_v].shape},{new_kv[layer_key][attn_type][k_v].shape}")
                     # print(f"接收到的 KV 缓存形状2: {target[layer_key][attn_type][k_v].shape},{new_kv[layer_key][attn_type][k_v].shape}")
          
      
     
    def continue_translation(self, job_data):
        self.model.decoder.open_kvcache()

        # 加载KV缓存
        for layer_idx, layer in enumerate(self.model.decoder.decoder_blocks):
            layer_key = f'layer_{layer_idx}'
            if layer_key in job_data['kv_cache']:
                layer.first_multihead_attn.kv_cache['K'] = job_data['kv_cache'][layer_key]['self_attn']['K']
                layer.first_multihead_attn.kv_cache['V'] = job_data['kv_cache'][layer_key]['self_attn']['V']
                layer.second_multihead_attn.kv_cache['K'] = job_data['kv_cache'][layer_key]['cross_attn']['K']
                layer.second_multihead_attn.kv_cache['V'] = job_data['kv_cache'][layer_key]['cross_attn']['V']
        
        #print(f"接收到的 KV 缓存形状1: {job_data['kv_cache'][layer_key]['self_attn']['K'].shape}")
        #print(f"接收到的 KV 缓存形状1: {job_data['kv_cache'][layer_key]['self_attn']['V'].shape}")
        
        en_token_ids = job_data.get('current_tokens', [BOS_IDX])
        while len(en_token_ids) < SEQ_MAX_LEN:
            dec_x = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z = self.model.decode(dec_x, job_data['encoder_z'], job_data['enc_x'])
            next_token_id = torch.argmax(decoder_z[0, -1]).item()
            en_token_ids.append(next_token_id)
            if next_token_id == EOS_IDX:
                if len(en_token_ids) >=  job_data['input_len']:
                    break

        self.model.decoder.close_kvcache()
        self.job_data = {'encoder_z': None, 'enc_x': None, 'kv_cache': {}}
        filtered = [id for id in en_token_ids if id not in [BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX]]
        return ' '.join(en_vocab.lookup_tokens(filtered))

if __name__ == '__main__':
    server = MigrationServer('checkpoints/model.pth')
    server.start()