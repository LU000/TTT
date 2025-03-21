# host_a.py
import torch
import socket
import pickle
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
import time
from dataset import de_preprocess,train_dataset,BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX,en_vocab,de_vocab
from utils.timer import SynchronizedWallClockTimer
Token_Qian = 'Token_Qian'
class MigrationClient:
    def __init__(self, host_b_addr):
        self.host_b_addr = host_b_addr
        self.migration_threshold = 3 # 100ms触发迁移
        self.timers = SynchronizedWallClockTimer()    
    def should_migrate(self, start_time):
        return (time.time() - start_time) > self.migration_threshold
    
    def send_migration_data(self, data):
        """发送迁移数据包"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.host_b_addr)
                serialized = pickle.dumps(data)
                #self.timers(Token_Qian).start()                
                # 协议头：MIGR标识 + 数据长度
                header = b'MIGR' + len(serialized).to_bytes(4, 'big')
                s.sendall(header + serialized)
                
   
                if s.recv(4) == b'ACK':
                    start_time2= data["start_time"]
                    recv_time2=time.time() 

                    #self.timers(Token_Qian).stop()
                    #self.gate_time = self.timers(Token_Qian)


                    #print(f"topk time: {self.gate_time}")                    
                    print(f"[HostB] 数据迁移Token时间: {recv_time2 - start_time2:.6f} 秒")   
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
 
    enc_x_batch=torch.tensor([de_ids],dtype=torch.long).to(DEVICE)     
    en_token_ids_total= de_ids.copy()
    # 解码初始化
    transformer.decoder.open_kvcache()
    en_token_ids = [BOS_IDX]
    migration_sent = False
    
    try:
        while len(en_token_ids) < SEQ_MAX_LEN:
            # 触发迁移检查
            if not migration_sent and client.should_migrate(start_time):
                print(f"en_token_ids_total={len(en_token_ids_total)}")
                migration_data = {
                    'en_token_ids_total': en_token_ids_total,
                    'start_time': time.time(),
                    'en_token_ids':en_token_ids,#记录了新生成的TokenID
                    'enc_x_batch': enc_x_batch.to(DEVICE),#历史TokenID
 
                }
 
                if client.send_migration_data(migration_data):
                    migration_sent = True
                    return "[迁移触发]"  # 中止本地生成
                
            # 正常解码
  
            if  en_token_ids == [BOS_IDX]:
                prefill = True
            else:
                prefill = False            
            
            dec_x_batch=torch.tensor([en_token_ids],dtype=torch.long).to(DEVICE)  # 准备decoder输入
            decoder_z=transformer.decode(dec_x_batch,enc_x_batch,prefill)#,page_id)   # decoder解碼
            next_token_probs=decoder_z[0,dec_x_batch.size(-1)-1,:]    # 序列下一个词的概率
            next_token_id=torch.argmax(next_token_probs)    # 下一个词ID
            en_token_ids.append(next_token_id)
            en_token_ids_total.append(next_token_id)
      
            if next_token_id==EOS_IDX:  # 结束符
                    break                
        # 后处理
        filtered = [id for id in en_token_ids 
                   if id not in [BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX]]
        return ' '.join(en_vocab.lookup_tokens(filtered))
    
    finally:
        transformer.decoder.close_kvcache()

if __name__ == '__main__':
    #start_time1=time.time()   
    transformer=Transformer(enc_vocab_size=len(de_vocab),dec_vocab_size=len(en_vocab),emb_size=512,q_k_size=64,v_size=64,f_size=2048,head=8,nblocks=6,dropout=0.1,seq_max_len=SEQ_MAX_LEN).to(DEVICE)
    transformer.load_state_dict(torch.load('checkpoints/model.pth'))
    transformer.eval()
    client = MigrationClient(('192.168.207.213', 12345))
     
    result = translate_with_migration(
        transformer, 
        'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.',
          client
    )
      
    text =  'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.'
      
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)