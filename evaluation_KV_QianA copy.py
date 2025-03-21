import torch
import socket
import pickle
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
import time
from dataset import de_preprocess, BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, en_vocab, de_vocab

class MigrationClient:
    def __init__(self, host_b_addr):
        self.host_b_addr = host_b_addr
        self.start_migration_time = 0.05
        self.stop_local_time = 0.2
        self.migrated_kv_positions = {}  # 记录各层KV缓存位置
        self.is_final = False

    def should_migrate(self, start_time):
        return (time.time() - start_time) > self.start_migration_time
    
    def should_stop_local(self, start_time):
        return (time.time() - start_time) > self.stop_local_time

    def send_migration_data(self, data, is_final,start_time):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.host_b_addr)
                serialized = pickle.dumps(data)
                header = b'MIGR' + len(serialized).to_bytes(4, 'big')
                s.sendall(header + serialized)

                if is_final:
                    s.sendall(b'DONE')
                    print("[HostA] 发送迁移完成信号'DONE'")

                if s.recv(4) == b'ACK':
                    print(f"{time.time() - start_time}")
                    print("[HostA] 迁移确认成功")
                return True
        except Exception as e:
            print(f"[HostA] 迁移错误: {e}")
            return False

def translate_with_migration(transformer, de_sentence, client):
    de_tokens, de_ids = de_preprocess(de_sentence)
    if len(de_tokens) > SEQ_MAX_LEN:
        raise Exception(f'句子长度超过{SEQ_MAX_LEN}')
 
    
    start_time = time.time()
    enc_x = torch.tensor([de_ids], dtype=torch.long).to(DEVICE)
    encoder_z = transformer.encode(enc_x)
    
    transformer.decoder.open_kvcache()
    en_token_ids = [BOS_IDX]
    migration_sent = False
    final_migration_sent = False
    input_len = len(de_tokens)
    print(f"{input_len}")
    try:
        while len(en_token_ids) < SEQ_MAX_LEN:
            current_time = time.time()
            
            # 首次迁移（0.1s前的KV）
            if not migration_sent and client.should_migrate(start_time):
                kv_cache = {}
                for i, layer in enumerate(transformer.decoder.decoder_blocks):
                    layer_key = f'layer_{i}'
                 #   print(layer_key)
                    kv_cache[layer_key] = {
                        'self_attn': {
                            'K': layer.first_multihead_attn.kv_cache['K'].clone().cpu(),
                            'V': layer.first_multihead_attn.kv_cache['V'].clone().cpu()
                        },
                        'cross_attn': {
                            'K': layer.second_multihead_attn.kv_cache['K'].clone().cpu(),
                            'V': layer.second_multihead_attn.kv_cache['V'].clone().cpu()
                        }
                    }
                    # 记录各层KV位置
                    client.migrated_kv_positions[layer_key] = {
                        'self_attn_K': kv_cache[layer_key]['self_attn']['K'].shape[1],
                        'self_attn_V': kv_cache[layer_key]['self_attn']['V'].shape[1],
                        'cross_attn_K': kv_cache[layer_key]['cross_attn']['K'].shape[1],
                        'cross_attn_V': kv_cache[layer_key]['cross_attn']['V'].shape[1]
                    }
                    
                migration_data = {
                    'enc_x': enc_x.cpu(),
                    'encoder_z': encoder_z.cpu(),
                    'kv_cache': kv_cache,
                    'input_len':input_len
                }
                print(f"接收到的 KV 缓存形状: {migration_data['kv_cache'][layer_key]['self_attn']['K'].shape}")
                print(f"接收到的 KV 缓存形状: {migration_data['kv_cache'][layer_key]['self_attn']['V'].shape}")

               
                if client.send_migration_data(migration_data, False,start_time):
                    migration_sent = True 
                # 在截取 KV 缓存之前打印出 pos 的值
                print(f"迁移时记录的位置：{ client.migrated_kv_positions[layer_key]}")

 

            # 最终迁移（0.1s-0.2s的KV）
            if migration_sent and not final_migration_sent and client.should_stop_local(start_time):
                final_kv_cache = {}
                for i, layer in enumerate(transformer.decoder.decoder_blocks):
                    layer_key = f'layer_{i}'
                    pos = client.migrated_kv_positions[layer_key]

                    # 截取新增的KV部分
                    self_attn_K = layer.first_multihead_attn.kv_cache['K'][:, pos['self_attn_K']:, :].clone().cpu()
                    self_attn_V = layer.first_multihead_attn.kv_cache['V'][:, pos['self_attn_V']:, :].clone().cpu()
                    cross_attn_K = layer.second_multihead_attn.kv_cache['K'][:, pos['cross_attn_K']:, :].clone().cpu()
                    cross_attn_V = layer.second_multihead_attn.kv_cache['V'][:, pos['cross_attn_V']:, :].clone().cpu()

                    final_kv_cache[layer_key] = {
                        'self_attn': {'K': self_attn_K, 'V': self_attn_V},
                        'cross_attn': {'K': cross_attn_K, 'V': cross_attn_V}
                    }

                final_migration_data = {
                    'enc_x': enc_x.cpu(),
                    'encoder_z': encoder_z.cpu(),
                    'current_tokens': en_token_ids.copy(),
                    'kv_cache': final_kv_cache,
                    'input_len':input_len
                }
                print(f"接收到的 KV 缓存形状1: {final_migration_data['kv_cache'][layer_key]['self_attn']['K'].shape}")
                print(f"接收到的 KV 缓存形状1: {final_migration_data['kv_cache'][layer_key]['self_attn']['V'].shape}")

              #  print(final_migration_data)
                
                if client.send_migration_data(final_migration_data, True,start_time):
                    final_migration_sent = True
                    print("[HostA] 终止本地计算，迁移增量KV")
                    return "迁移完成，HostB继续推理"
                

            # 正常解码

            dec_x = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z = transformer.decode(dec_x, encoder_z, enc_x)
            next_token_id = torch.argmax(decoder_z[0, -1, :]).item()
            en_token_ids.append(next_token_id)
            print(en_token_ids)

            if next_token_id == EOS_IDX:
                if len(en_token_ids) >= len(de_tokens):
                    break
     
        filtered = [id for id in en_token_ids if id not in [BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX]]
        return ' '.join(en_vocab.lookup_tokens(filtered))
    
    finally:
        transformer.decoder.close_kvcache()

if __name__ == '__main__':
    transformer = Transformer(enc_vocab_size=len(de_vocab), dec_vocab_size=len(en_vocab), 
                            emb_size=512, q_k_size=64, v_size=64, f_size=2048, 
                            head=8, nblocks=6, dropout=0.1, seq_max_len=SEQ_MAX_LEN).to(DEVICE)
    transformer.load_state_dict(torch.load('checkpoints/model.pth'))
    transformer.eval()
    client = MigrationClient(('192.168.207.213', 12345))
    
    
    result = translate_with_migration(
        transformer, 
        'Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen ParkZwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen ParkZwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park.',
         client
    )
      
    text = 'Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen ParkZwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen ParkZwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park.',
      
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)