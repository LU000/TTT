# host_a.py
import torch
import socket
import pickle
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
import time
from dataset import de_preprocess, en_vocab, de_vocab, BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX

class MigrationClient:
    def __init__(self, host_b_addr):
        self.host_b_addr = host_b_addr
        self.migration_threshold = 0.1 # 100ms触发迁移
        self.migration_ratio = 0.5

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
                    start_time2= data["start_time"]
                    recv_time2=time.time() 
                    print(f"[HostB] 数据迁移KV时间: {recv_time2 - start_time2:.6f} 秒")    
                
                    return True
        except Exception as e:
            print(f"[HostA] 迁移失败: {str(e)}")
            return False

def translate_with_migration(transformer, de_sentence, client):
    de_tokens, de_ids = de_preprocess(de_sentence)
    enc_x = torch.tensor([de_ids], dtype=torch.long).to(DEVICE)
    encoder_z = transformer.encode(enc_x)
    
 
    transformer.decoder.open_kvcache()
    en_token_ids = [BOS_IDX]
    start_time=time.time()
    try:
         
        while len(en_token_ids) < SEQ_MAX_LEN:
            if client.should_migrate(start_time):
                total_len = len(en_token_ids)
                split_idx = int(total_len * client.migration_ratio)
                
                # 分割Token序列
                token_part = en_token_ids[:split_idx]
                kv_part_len = total_len - split_idx
                
                # 构建迁移数据包
                migration_data = {
                    'enc_x': enc_x.cpu(),
                    'token_part': token_part,
                    'kv_part': {
                        'start_idx': split_idx,
                        'cache': {}
                    },
                    'start_time': time.time(),
                    'current_tokens':en_token_ids

                }
                
                # 提取后半部分KV缓存
                for layer_idx, layer in enumerate(transformer.decoder.decoder_blocks):
                    layer_cache = {}
                    
                    # 自注意力KV截取
                    self_k = layer.first_multihead_attn.kv_cache['K'][:, split_idx:, :]
                    self_v = layer.first_multihead_attn.kv_cache['V'][:, split_idx:, :]
                    layer_cache['self_attn'] = {'K': self_k.cpu(), 'V': self_v.cpu()}
                    
                    # 交叉注意力KV截取
                    cross_k = layer.second_multihead_attn.kv_cache['K'][:, split_idx:, :]
                    cross_v = layer.second_multihead_attn.kv_cache['V'][:, split_idx:, :]
                    layer_cache['cross_attn'] = {'K': cross_k.cpu(), 'V': cross_v.cpu()}
                    
                    migration_data['kv_part']['cache'][f'layer_{layer_idx}'] = layer_cache
                
                if client.send_migration_data(migration_data):
                    return "[迁移触发]"
                
            # 正常生成流程
            dec_input = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z = transformer.decode(dec_input, encoder_z, enc_x)
            en_token_ids.append(torch.argmax(decoder_z[0, -1]).item())
            
    finally:
        transformer.decoder.close_kvcache()

if __name__ == '__main__':
   # start_time1=time.time()
    transformer = Transformer(
        enc_vocab_size=len(de_vocab), dec_vocab_size=len(en_vocab),
        emb_size=512, q_k_size=64, v_size=64, f_size=2048,
        head=8, nblocks=6, dropout=0.1, seq_max_len=SEQ_MAX_LEN
    ).to(DEVICE)
    transformer.load_state_dict(torch.load('checkpoints/model.pth'))
    transformer.eval()
    client = MigrationClient(('192.168.207.213', 12345))
    
    result = translate_with_migration(
        transformer, 
        'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.',
         client
    )
   # end_time1=time.time()   
   
    text ='Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.'
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)

    '''
    result = translate_with_migration(
        transformer, 
        'Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park. Ein Kind spielt mit einem Ball, während ein Hund daneben sitzt. Die Sonne scheint hell und die Vögel singen in den Bäumen. Eine Frau liest ein Buch auf einer Bank, während ein Mann telefoniert. Am Horizont sieht man hohe Berge und einen blauen Himmel. Ein kleines Mädchen hält einen roten Luftballon und lacht. Die Straßen sind voller Menschen, die einkaufen oder spazieren gehen. Ein alter Mann füttert Tauben auf dem Marktplatz. Ein Junge fährt mit seinem Fahrrad schnell die Straße hinunter. Neben ihm läuft sein Hund und bellt fröhlich. Ein Auto hält an der Ampel, während Fußgänger die Straße überqueren. In einem Café sitzen Freunde zusammen und trinken Kaffee. Ein Kellner bringt ihnen frische Croissants. Die Stadt ist lebendig und voller Geräusche. Eine Gruppe von Touristen macht Fotos von einem alten Gebäude. Eine Mutter schiebt einen Kinderwagen und spricht mit einer Freundin. Der Wind weht sanft durch die Blätter der Bäume. Eine Katze sitzt auf einem Fensterbrett und beobachtet die Menschen unten. Die Glocken einer Kirche läuten zur vollen Stunde. Ein Straßenmusiker spielt eine Melodie auf seiner Gitarre. Die Menschen bleiben stehen und hören zu. Ein kleines Kind klatscht begeistert in die Hände. Eine Straßenbahn fährt vorbei und bringt die Menschen zu ihren Zielen. In einem Park machen einige Leute Yoga auf einer grünen Wiese. Ein Mann liest die Zeitung, während er einen Kaffee trinkt. Ein Obdachloser sitzt mit seinem Hund an einer Straßenecke. Die Lichter der Stadt beginnen zu leuchten, während die Sonne untergeht. Ein Paar hält Händchen und genießt den Abend. Die Straßenlaternen werfen lange Schatten auf das Kopfsteinpflaster. Eine Gruppe von Jugendlichen lacht und macht Witze. Ein Künstler malt ein Bild von der Stadt auf einer Leinwand. Die Nacht bricht herein, aber die Stadt bleibt wach. Menschen tanzen in einem Club zur Musik. Ein Taxifahrer wartet auf Fahrgäste an der Straßenecke. Die Fenster der Gebäude leuchten in warmem Licht. Eine Frau schaut aus dem Fenster und denkt nach. Ein Mann joggt am Fluss entlang, während die Stadt schläft. Der Mond scheint hell am Himmel, und die Sterne funkeln. Die Geräusche der Nacht sind leiser, aber die Stadt lebt weiter.',
         client
    )
      
    text ='Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park. Ein Kind spielt mit einem Ball, während ein Hund daneben sitzt. Die Sonne scheint hell und die Vögel singen in den Bäumen. Eine Frau liest ein Buch auf einer Bank, während ein Mann telefoniert. Am Horizont sieht man hohe Berge und einen blauen Himmel. Ein kleines Mädchen hält einen roten Luftballon und lacht. Die Straßen sind voller Menschen, die einkaufen oder spazieren gehen. Ein alter Mann füttert Tauben auf dem Marktplatz. Ein Junge fährt mit seinem Fahrrad schnell die Straße hinunter. Neben ihm läuft sein Hund und bellt fröhlich. Ein Auto hält an der Ampel, während Fußgänger die Straße überqueren. In einem Café sitzen Freunde zusammen und trinken Kaffee. Ein Kellner bringt ihnen frische Croissants. Die Stadt ist lebendig und voller Geräusche. Eine Gruppe von Touristen macht Fotos von einem alten Gebäude. Eine Mutter schiebt einen Kinderwagen und spricht mit einer Freundin. Der Wind weht sanft durch die Blätter der Bäume. Eine Katze sitzt auf einem Fensterbrett und beobachtet die Menschen unten. Die Glocken einer Kirche läuten zur vollen Stunde. Ein Straßenmusiker spielt eine Melodie auf seiner Gitarre. Die Menschen bleiben stehen und hören zu. Ein kleines Kind klatscht begeistert in die Hände. Eine Straßenbahn fährt vorbei und bringt die Menschen zu ihren Zielen. In einem Park machen einige Leute Yoga auf einer grünen Wiese. Ein Mann liest die Zeitung, während er einen Kaffee trinkt. Ein Obdachloser sitzt mit seinem Hund an einer Straßenecke. Die Lichter der Stadt beginnen zu leuchten, während die Sonne untergeht. Ein Paar hält Händchen und genießt den Abend. Die Straßenlaternen werfen lange Schatten auf das Kopfsteinpflaster. Eine Gruppe von Jugendlichen lacht und macht Witze. Ein Künstler malt ein Bild von der Stadt auf einer Leinwand. Die Nacht bricht herein, aber die Stadt bleibt wach. Menschen tanzen in einem Club zur Musik. Ein Taxifahrer wartet auf Fahrgäste an der Straßenecke. Die Fenster der Gebäude leuchten in warmem Licht. Eine Frau schaut aus dem Fenster und denkt nach. Ein Mann joggt am Fluss entlang, während die Stadt schläft. Der Mond scheint hell am Himmel, und die Sterne funkeln. Die Geräusche der Nacht sind leiser, aber die Stadt lebt weiter. '
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)
    '''