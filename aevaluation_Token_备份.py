# host_a.py
import torch
import socket
import pickle
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
import time
from dataset import de_preprocess,train_dataset,BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX,en_vocab,de_vocab

class MigrationClient:
    def __init__(self, host_b_addr):
        self.host_b_addr = host_b_addr
        self.migration_threshold = 0.1  # 100ms触发迁移
    
    def should_migrate(self, start_time):
        return (time.time() - start_time) > self.migration_threshold
    
    def send_migration_data(self, data):
        """发送迁移数据包"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.host_b_addr)
                serialized = pickle.dumps(data)
                
                # 协议头：MIGR标识 + 数据长度
                header = b'MIGR' + len(serialized).to_bytes(4, 'big')
                s.sendall(header + serialized)
                
                # 等待确认
                if s.recv(4) == b'ACK':
                    print("[HostA] 迁移成功")
                    return True
        except Exception as e:
            print(f"[HostA] 迁移失败: {str(e)}")
            return False

def translate_with_migration(transformer, de_sentence, client):
    """支持迁移的翻译函数"""
     # De分词
    de_tokens,de_ids=de_preprocess(de_sentence)
    if len(de_tokens)>SEQ_MAX_LEN:
        raise Exception('不支持超过{}的句子'.format(SEQ_MAX_LEN))
    
    start_time = time.time()
    # Encoder阶段
    enc_x=torch.tensor([de_ids],dtype=torch.long).to(DEVICE)      # 准备encoder输入
    encoder_z=transformer.encode(enc_x)    # encoder编码
    
 
    # 解码初始化
    transformer.decoder.open_kvcache()
    en_token_ids = [BOS_IDX]
    migration_sent = False
    
    try:
        while len(en_token_ids) < SEQ_MAX_LEN:
            # 触发迁移检查
            if not migration_sent and client.should_migrate(start_time):
                
                migration_data = {
                    'enc_x':enc_x.cpu(),
                    #'encoder_z': encoder_z.cpu(),
                    'current_tokens': en_token_ids.copy(),
                    'start_time': time.time(),
                    #'kv_cache': {
                    #    f'layer_{i}': {
                            # 自注意力的缓存（新增）
                    #       'self_attn': {
                    #          'K': layer.first_multihead_attn.kv_cache['K'].cpu(),
                    #          'V': layer.first_multihead_attn.kv_cache['V'].cpu()
                    #        },
                    #        # 交叉注意力的缓存（原有）
                    #        'cross_attn': {
                     #          'K': layer.second_multihead_attn.kv_cache['K'].cpu(),
                    #           'V': layer.second_multihead_attn.kv_cache['V'].cpu()
                    #}
                     #    } for i, layer in enumerate(transformer.decoder.decoder_blocks)
                     #}
                }
                # 添加形状校验
                #print(f'[HostA]enc_x:{enc_x.shape}, encoder_z, 形状: {encoder_z.shape}')
                #for i, layer in enumerate(transformer.decoder.decoder_blocks):
                #      print(f"[HostA] 层 {i} 交叉注意力缓存形状: K={layer.second_multihead_attn.kv_cache['K'].shape}")
                #      print(f"[HostA] 层 {i} 自注意力缓存形状: K={layer.first_multihead_attn.kv_cache['K'].shape}")
                # 添加形状日志
                #print(f"[HostA] encoder_z 形状: {encoder_z.shape}")
                # for i, layer in enumerate(transformer.decoder.decoder_blocks):
                #    print(f"[HostA] 层 {i} 交叉注意力缓存形状: K={layer.second_multihead_attn.kv_cache['K'].shape}")

                #print(f'migration_data={migration_data}')
                #client.sendall(header)
                if client.send_migration_data(migration_data):
                    migration_sent = True
                    return "[迁移触发]"  # 中止本地生成
                
            # 正常解码
            dec_x = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z = transformer.decode(dec_x, encoder_z, enc_x)
            
            next_token_id = torch.argmax(decoder_z[0, -1, :]).item()
            en_token_ids.append(next_token_id)
            
            if next_token_id == EOS_IDX:
                break
                
        # 后处理
        filtered = [id for id in en_token_ids 
                   if id not in [BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX]]
        return ' '.join(en_vocab.lookup_tokens(filtered))
    
    finally:
        transformer.decoder.close_kvcache()

if __name__ == '__main__':
    start_time1=time.time()   
    transformer=Transformer(enc_vocab_size=len(de_vocab),dec_vocab_size=len(en_vocab),emb_size=512,q_k_size=64,v_size=64,f_size=2048,head=8,nblocks=6,dropout=0.1,seq_max_len=SEQ_MAX_LEN).to(DEVICE)
    transformer.load_state_dict(torch.load('checkpoints/model.pth'))
    transformer.eval()
    client = MigrationClient(('192.168.207.213', 12345))
     
    result = translate_with_migration(
        transformer, 
        'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.',
         client
    )
    end_time1=time.time()   
    print(f"[HostB] 数据迁移时间: {end_time1 - start_time1:.6f} 秒")       
    text ='Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.'
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)