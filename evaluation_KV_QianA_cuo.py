import torch
import socket
import pickle
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataset import de_preprocess, train_dataset,BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, en_vocab, de_vocab
BANDWIDTH = 100000  # 模拟100Mbps带宽

class ThreadSafeMigrationClient:
    def __init__(self, host_b_addr):
        self.host_b_addr = host_b_addr
        self.start_migration_time = 10
        self.migrated_kv_positions = {}
        self.received_ack = False
        self.initial_kv_positions = None
        self.chunk_size = 193074   
        self.ack_lock = threading.Lock()
       
        self.current_page = {}  # 当前活跃页 {request_id: page_id}
        self.page_counter = 0
                # 使用线程本地存储
        self._local = threading.local()
 
    def get_new_page_id(self):
        with self.ack_lock:
            self.page_counter += 1
            return f"page_{self.page_counter}" 
    
    def should_migrate(self, start_time):
        return (time.time() - start_time) > self.start_migration_time
    
    def _simulate_network_delay(self, data_size):
        """根据数据大小和带宽计算模拟延迟"""
        bits = data_size * 8  # 转换为比特
        delay = bits / (BANDWIDTH * 1e6)  # 带宽单位换算为bps
        time.sleep(delay)

    def send_migration_data_async(self, data, start_time):
        def migration_thread():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(self.host_b_addr)
 
                    serialized = pickle.dumps(data)
                    header = b'MIGR' + len(serialized).to_bytes(4, 'big')
                    # 修改打印逻辑：
                    print(f"数据体积: {len(serialized)} 字节")  # 替代 len(data)
                    #s.sendall(header + serialized)
                    s.sendall(header)  # 先发送头
                    print(f"{len(data)} ")
                    # 分块发送数据并计算延迟
                    total_sent = 0
                    while total_sent < len(serialized):
                        chunk = serialized[total_sent:total_sent+self.chunk_size]
                        s.sendall(chunk)
                        total_sent += len(chunk)
                        self._simulate_network_delay(len(chunk))  # 对每个块模拟延迟

                    if s.recv(4) == b'ACK':
                        with self.ack_lock:
                            self.received_ack = True
                        print(f"[HostA] ACK接收时间: {time.time() - start_time:.4f}s")
            except Exception as e:
                print(f"[HostA] 迁移错误: {e}")

        threading.Thread(target=migration_thread, daemon=True).start()

def translate_with_migration(transformer, de_sentence, client, page_id):
    de_tokens, de_ids = de_preprocess(de_sentence)
    if len(de_tokens) > SEQ_MAX_LEN:
        raise Exception(f'句子长度超过{SEQ_MAX_LEN}')
    
    #page_id = client.get_new_page_id()
  

    # 动态判断请求类型
    is_long_request = len(de_tokens) > 100  # 假设50个token为阈值
    
    # 公共预处理
    start_time = time.time()
    enc_x = torch.tensor([de_ids], dtype=torch.long).to(DEVICE)
    encoder_z = transformer.encode(enc_x)
    
    if is_long_request:
        # 长请求迁移处理
        return handle_long_request(transformer, enc_x, encoder_z, de_tokens, client, start_time,page_id)
    else:
        # 短请求本地处理
        return handle_short_request(transformer, enc_x, encoder_z, page_id)


def handle_long_request(transformer, enc_x, encoder_z, de_tokens, client, start_time, page_id):  
    transformer.decoder.open_kvcache()
    en_token_ids = [BOS_IDX]
    migration_sent = False
    input_len = len(de_tokens)
    print(f"page_id={page_id}")
    try:
   
        while len(en_token_ids) < SEQ_MAX_LEN  :
            # 触发首次迁移
            #print(f"{time.time()-start_time}")
            if not migration_sent and client.should_migrate(start_time):
                client.initial_kv_positions = {}
                initial_kv_cache = {}
 
                # 记录初始KV缓存和位置
                for i, layer in enumerate(transformer.decoder.decoder_blocks):
                    layer_key = f'layer_{i}'

                    print(f"{layer.first_multihead_attn.get_kv(page_id, layer_id=i)['K'].shape[1] }")
                    print(f"{layer.first_multihead_attn.get_kv(page_id, layer_id=i)['V'].shape[1] }")

                    client.initial_kv_positions[layer_key] = {    
                        'self_attn_K': layer.first_multihead_attn.get_kv(page_id, layer_id=i)['K'].shape[1],
                        'self_attn_V': layer.first_multihead_attn.get_kv(page_id, layer_id=i)['V'].shape[1],
                    }
                    
                    print(f"{client.initial_kv_positions[layer_key] }")
                    
                    initial_kv_cache[layer_key] = {
                        'self_attn': {
                            'K': layer.first_multihead_attn.get_kv(page_id, layer_id=i)['K'].clone().cpu(),
                            'V': layer.first_multihead_attn.get_kv(page_id, layer_id=i)['V'].clone().cpu()
                        },
                        'cross_attn': {
                            'K': layer.second_multihead_attn.get_kv(page_id, layer_id=i)['K'].clone().cpu(),
                            'V': layer.second_multihead_attn.get_kv(page_id, layer_id=i)['V'].clone().cpu()
                        }
                    }
            
                               
                print(f"生成的 KV 缓存形状: {initial_kv_cache[layer_key]['self_attn']['K'].shape}")
                print(f"生成的 KV 缓存形状: {initial_kv_cache[layer_key]['self_attn']['V'].shape}")

              
                # 异步发送初始迁移数据
                client.send_migration_data_async({
                    'enc_x': enc_x.cpu(),
                    'encoder_z': encoder_z.cpu(),
                    'kv_cache': initial_kv_cache,
                    'initial_positions': client.initial_kv_positions,
                    'input_len':input_len
                }, start_time)
                migration_sent = True
                print(f"[HostA] 初始迁移已发送 ({time.time()-start_time:.4f}s)")

            # 检查ACK状态
            if migration_sent and client.received_ack:
                # 收集增量KV缓存
                delta_kv_cache = {}
                for i, layer in enumerate(transformer.decoder.decoder_blocks):
                    layer_key = f'layer_{i}'
                    initial_pos = client.initial_kv_positions[layer_key]
                   # print(f"{initial_pos}")
                    delta_kv_cache[layer_key] = {
                        'self_attn': {
                            'K': layer.first_multihead_attn.get_kv(page_id, layer_id=i)['K'][:, initial_pos['self_attn_K']:, :].clone().cpu(),
                            'V': layer.first_multihead_attn.get_kv(page_id, layer_id=i)['V'][:, initial_pos['self_attn_V']:, :].clone().cpu()
                        },
                        'cross_attn': {
                            'K': layer.second_multihead_attn.get_kv(page_id, layer_id=i)['K'][:, initial_pos['cross_attn_K']:, :].clone().cpu(),
                            'V': layer.second_multihead_attn.get_kv(page_id, layer_id=i)['V'][:, initial_pos['cross_attn_V']:, :].clone().cpu()
                        }
                    }
                                        
                print(f"接收到的 KV 缓存形状1: {delta_kv_cache[layer_key]['self_attn']['K'].shape}")
                print(f"接收到的 KV 缓存形状1: {delta_kv_cache[layer_key]['self_attn']['V'].shape}")

                # 同步发送最终迁移数据
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(client.host_b_addr)
                    final_data = {
                        'current_tokens': en_token_ids.copy(),
                        'kv_cache': delta_kv_cache,
                        'is_final': True,
                        'input_len':input_len                        
                    }
                    serialized = pickle.dumps(final_data)
                    header = b'MIGR' + len(serialized).to_bytes(4, 'big')
                    #s.sendall(header + serialized)
                    s.sendall(header)
                    print(f"数据体积: {len(serialized)} 字节")  # 替代 len(data)
                    # 分块发送数据体
                    total_sent = 0
                    print(f"[HostA] 增量KV已发送 ({time.time()-start_time:.4f}s)")
                    while total_sent < len(serialized):
                        chunk = serialized[total_sent:total_sent+client.chunk_size]
                        s.sendall(chunk)
                        total_sent += len(chunk)
                        client._simulate_network_delay(len(chunk))  # 模拟带宽

                  
                    if s.recv(4) == b'ACK':
                        print(f"[HostA] 增量KV ACK接收时间: {time.time() - start_time:.4f}s")
 
                return "迁移完成,HostB继续推理"

            # 正常解码流程
            dec_x = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z = transformer.decode(dec_x, encoder_z, enc_x,page_id)
            next_token_id = torch.argmax(decoder_z[0, -1, :]).item()
            en_token_ids.append(next_token_id)
           # print(f"{en_token_ids}")
            time.sleep(0.5)
            if next_token_id == EOS_IDX:
               if len(en_token_ids) >= len(de_tokens):
                 break
        return process_final_result(en_token_ids)
    
    finally:
        transformer.decoder.close_kvcache()
     

def handle_short_request(transformer, enc_x, encoder_z, page_id):
    transformer.decoder.open_kvcache()
    en_token_ids = [BOS_IDX]

    try:
        while len(en_token_ids) < SEQ_MAX_LEN:
            dec_x = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z = transformer.decode(dec_x, encoder_z, enc_x,page_id)
            next_token_id = torch.argmax(decoder_z[0, -1, :]).item()
            en_token_ids.append(next_token_id)
            if next_token_id == EOS_IDX:
                break
        return process_final_result(en_token_ids)
    finally:
        transformer.decoder.close_kvcache()
     

def process_final_result(en_token_ids):
    filtered = [id for id in en_token_ids if id not in [BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX]]
    return ' '.join(en_vocab.lookup_tokens(filtered))
 

if __name__ == '__main__':
  import concurrent.futures
  import time
  
  transformer = Transformer(enc_vocab_size=len(de_vocab), dec_vocab_size=len(en_vocab), 
                            emb_size=512, q_k_size=64, v_size=64, f_size=2048, 
                            head=8, nblocks=6, dropout=0.1, seq_max_len=SEQ_MAX_LEN).to(DEVICE)

  transformer.load_state_dict(torch.load('checkpoints/model.pth'))
  transformer.eval()

  def simulate_requests(transformer, num_short_requests):
    client = ThreadSafeMigrationClient(('192.168.207.213', 12345))  # 需要替换为实际IP和端口

    # 长请求（单独启动）
    long_future = executor.submit(translate_with_migration, transformer, 'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.', client,100)

    # 记录短请求的开始时间
    short_request_times = []

    # 运行多个短请求
    short_futures = []
    for i in range(num_short_requests):
        short_futures.append(executor.submit(translate_with_migration, transformer , "Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.", client, i+1) )
        short_request_times.append(time.time())

    # 等待所有短请求完成
    for future in concurrent.futures.as_completed(short_futures):
        try:
            result = future.result()
            print(f"短请求处理完成: {result}")
        except Exception as e:
            print(f"短请求异常: {e}")

    # 计算短请求的平均处理时间
    avg_short_request_time = sum(time.time() - t for t in short_request_times) / num_short_requests
    print(f"平均短请求处理时间: {avg_short_request_time:.4f} 秒")

    # 等待长请求完成
    try:
        long_result = long_future.result()
        print(f"长请求处理完成: {long_result}")
    except Exception as e:
       print(f"长请求异常: {e}")

   # 线程池
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for num_requests in range(1, 4):  # 测试1到20个短请求的负载
        print(f"\n测试 {num_requests} 个短请求:")
        simulate_requests(transformer, num_requests)

 