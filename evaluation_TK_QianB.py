# host_b.py
import torch
import socket
import pickle
import threading
import time
import uuid
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
from dataset import BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, en_vocab, de_vocab
from utils.timer import SynchronizedWallClockTimer
Token_Qian = 'Token_Qian'

class HybridMigrationServer:
    def __init__(self, model_path):
        self.model = Transformer(enc_vocab_size=len(de_vocab),dec_vocab_size=len(en_vocab),emb_size=512,q_k_size=64,v_size=64,f_size=2048,head=8,nblocks=6,dropout=0.1,seq_max_len=SEQ_MAX_LEN).to(DEVICE) 
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        
        self.current_jobs = {}
        self.kv_buffer = {}
        self.lock = threading.Lock()
        self._init_kv_cache()
        self.active_connections = {}
        self.timers = SynchronizedWallClockTimer()    

    def _init_kv_cache(self):
        """初始化所有层的KV缓存结构"""
        for layer in self.model.decoder.decoder_blocks:
            layer.self_attn.kv_cache = {'K': torch.tensor([]), 'V': torch.tensor([])}

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
            #print(f"{len(data),length}")
    
        return pickle.loads(data)

    def handle_connection(self, conn, addr):
        try:
            starttime=time.time()
            # 生成唯一迁移ID
            migration_id = str(uuid.uuid4())
            
            # 接收Token部分
            header = conn.recv(4)
            if header != b'Tokn':
                print(f"[HostB] 无效协议头: {header}")
                return

            token_length = int.from_bytes(conn.recv(4), 'big')
            #print(f"{token_length}")
            token_data = self._receive_all(conn, token_length)
            conn.send(b'ACK')  # 确认Token接收
            #print(f'token_data={token_data}')
            # 存储临时数据
            with self.lock:
                self.active_connections[migration_id] = {
                    'conn': conn,
                    'token_data': token_data,
                    'en_token_ids': None,
                    'enc_x_batch': None
                }

            self.timers(Token_Qian).start()   
            start_time2=time.time()
            # 处理Token数据（同步执行）
            self.process_token_data(migration_id)
            end_time2=time.time()
            self.timers(Token_Qian).stop()
            self.gate_time = self.timers(Token_Qian).elapsed(reset=False)
            print(f"topk time: {self.gate_time}")    

            print(f"[HostB] Token处理时间: {end_time2 - start_time2:.6f} 秒")  
            # 启动KV接收线程
            kv_thread = threading.Thread(
                target=self._receive_kv_data,
                args=(conn, migration_id)
            )             
            kv_thread.start()
   

          
            # 等待KV接收完成或超时
            kv_thread.join(timeout=5)
            if kv_thread.is_alive():
                 raise TimeoutError("KV数据接收超时")     
            
            with self.lock:
                data = self.active_connections[migration_id]
                if 'kv_data' not in data:
                    raise ValueError("未收到完整的KV数据")


            #print("1")
            # 合并KV缓存并生成结果
            result = self.merge_and_generate(migration_id)
            #print("2")
            print(f"[HostB] 任务完成: {result}")  
            endtime=time.time()
            print(f"完成时间={endtime-starttime}")  
            return result
        
        except Exception as e:
            print(f"[HostB] 处理错误: {str(e)}")
        finally:
            with self.lock:
                if migration_id in self.active_connections:
                    del self.active_connections[migration_id]
            conn.close()

    def _receive_kv_data(self, conn, migration_id):
        """后台接收KV数据的线程"""
        try:
            header = conn.recv(4)
            if header != b'KVKV':
                print(f"[HostB] 无效KV协议头: {header}")
                return
            kv_length = int.from_bytes(conn.recv(4), 'big')
            kv_data = self._receive_all(conn, kv_length)
            recv_time = time.time()
            start_time = kv_data['start_time']
            print(f"[HostB] 数据迁移KV Cache时间: {recv_time - start_time:.6f} 秒")    
     
            #print(f'kv_data={kv_data}')
            with self.lock:
                if migration_id in self.active_connections:
                    self.active_connections[migration_id]['kv_data'] = kv_data
                    conn.send(b'ACK')
        except Exception as e:
            print(f"[HostB] KV接收失败: {str(e)}")

    def process_token_data(self, migration_id):
       
        """处理Token部分并生成初始KV缓存"""
        with self.lock:
            data = self.active_connections[migration_id]
            token_data = data['token_data']  # 先获取token_data字典
        # 从token_data中提取字段
        token_part = token_data['token_part'] 
        enc_x_batch = token_data['enc_x_batch'].to(DEVICE)     # 注意：此处应来自token_data
        #en_token_ids = token_data['en_token_ids']   
        #print("3")  
        # 初始化解码器
        self.model.decoder.open_kvcache()
        start_time1=time.time()        
        print(f'[HostB]enc_x_batch:{len(enc_x_batch)}, token_part, 形状: {len(token_part)}')
        #print(f'len={len(en_token_ids)}') 
        if len(token_part) > 1:
           #print(f'{en_token_ids[:-1]}')
           dec_x_full = torch.tensor([token_part], dtype=torch.long).to(DEVICE)
          # print(f'{dec_x_full[:-1]}')
           _ = self.model.decode(dec_x_full, enc_x_batch, prefill=True)

        end_time1=time.time()
        print(f"[HostB] Token处理时间: {end_time1 - start_time1:.6f} 秒")    
     
        # 保存关键数据
        with self.lock:
            self.active_connections[migration_id].update({
                'en_token_ids': token_data['en_token_ids'],
                'enc_x_batch': token_data['enc_x_batch'],
            })

    def merge_and_generate(self, migration_id):
        """合并KV缓存并生成剩余内容"""
        with self.lock:
            data = self.active_connections.get(migration_id)
            if not data:
                return "[错误] 迁移任务已丢失"
        
            # 获取必要数据
            enc_x_batch = data['enc_x_batch'].to(DEVICE) 
            en_token_ids = data['en_token_ids']
           # print(f"en_token_ids={len(en_token_ids)},enc_x_batch={enc_x_batch}")
            kv_data = data.get('kv_data')
        #print("4") 
        #print(f'current_tokens={current_tokens}')           
        #print(f'kv_data={kv_data}')    
        
        # 阶段2：拼接迁移的KV缓存
        for layer_idx in range(len(self.model.decoder.decoder_blocks)):
            layer = self.model.decoder.decoder_blocks[layer_idx]
            migrated_cache = kv_data['kv_part']['cache'][f'layer_{layer_idx}']
            #print(f"{kv_data['kv_part']['cache'][f'layer_{layer_idx}']}")
            #print(f'migrated_cache={migrated_cache}')
            # 自注意力拼接
            #print(migrated_cache['self_attn']['K'].shape)
            self_attn = layer.self_attn
            self._concat_kv(
                self_attn.kv_cache, 
                migrated_cache['self_attn'],
                kv_data['kv_part']['start_idx']
            )
            
        # 阶段3：继续生成
        
        return self._generate_remaining(enc_x_batch, en_token_ids)

    def _concat_kv(self, target_cache, migrated_cache, start_idx):
        """动态KV缓存合并"""
        for key in ['K', 'V']:
            original = target_cache[key]
            migrated = migrated_cache[key].to(DEVICE)
            #print(f'original={original}')
            #print(f'migrated={migrated}')
            # 维度对齐处理
            if original.size(1) < start_idx:
                padding = torch.zeros(
                    original.size(0),
                    start_idx - original.size(1),
                    original.size(2),
                    device=DEVICE
                )
                original = torch.cat([original, padding], dim=1)
            
            # 拼接新缓存
            target_cache[key] = torch.cat([
                original[:, :start_idx, :],
                migrated
            ], dim=1)
            

    def _generate_remaining(self, enc_x_batch,en_token_ids):
        """生成剩余Token"""
        try:
            while len(en_token_ids)<SEQ_MAX_LEN:
                if  en_token_ids == [BOS_IDX]:
                    prefill = True
                else:
                    prefill = False                 
                dec_x_batch=torch.tensor([en_token_ids],dtype=torch.long).to(DEVICE)  # 准备decoder输入
                decoder_z=self.model.decode(dec_x_batch,enc_x_batch,prefill)   # decoder解碼
                next_token_probs=decoder_z[0,dec_x_batch.size(-1)-1,:]    # 序列下一个词的概率
                next_token_id=torch.argmax(next_token_probs)    # 下一个词ID
                en_token_ids.append(next_token_id)
                #print(f'next_token_id={next_token_id}')
                if next_token_id==EOS_IDX:  # 
                    #if len(en_token_ids) >=  input_len: 
                          break
            self.model.decoder.close_kvcache() # 清理KV Cache
            #print(f'en_token_ids={en_token_ids}')
             # 生成翻译结果
            #print(f'en_token_ids={en_token_ids},enc_x_batch={enc_x_batch}')
            en_token_ids=[id for id in en_token_ids if id not in [BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX]] # 忽略特殊字符
            en_tokens=en_vocab.lookup_tokens(en_token_ids)    # 词id序列转token序列
             
            return ' '.join(en_tokens)
        
        finally:
            self.model.decoder.close_kvcache()
        
 
if __name__ == '__main__': 

    server = HybridMigrationServer('checkpoints/model.pth')
    server.start() 
