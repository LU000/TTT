# host_a.py
import torch
import socket
import pickle
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
import time
from dataset import de_preprocess, en_vocab, de_vocab, BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX
BANDWIDTH = 8 * 1e9 

class MigrationClient:
    def __init__(self, host_b_addr):
        self.host_b_addr = host_b_addr
        self.migration_threshold = 0.1 # 100ms触发迁移
        self.migration_ratio = 0.98
        self.chunk_size = 10485760000  

    def _simulate_network_delay(self, data_size):
       bits = data_size * 8  # 转换为比特
       expected_time = bits / (BANDWIDTH * 1e6)  # 计算理论传输时间
  
       start_time = time.time()
       while time.time() - start_time < expected_time:
           time.sleep(0.001)  # 小步暂停，避免卡死
        
    def should_migrate(self, start_time):
        return (time.time() - start_time) > self.migration_threshold
    
    def send_migration_data(self, token_data, kv_data):
        """分两次发送迁移数据：先Token，后KV"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.host_b_addr)
                start_time1=time.time() 
                # 发送Token部分
                serialized_token = pickle.dumps(token_data)
                header_token = b'Tokn' + len(serialized_token).to_bytes(4, 'big')
                s.sendall(header_token)
                
                print(f"serialized_token={len(serialized_token)}")
                total_sent = 0
                while total_sent < len(serialized_token):
                    chunk = serialized_token[total_sent:total_sent+self.chunk_size]
                    s.sendall(chunk)
                    total_sent += len(chunk)
                   
                    #print(f"{total_sent, len(serialized_token)}")
                    #self._simulate_network_delay(len(chunk))  # 对每个块模拟延迟

                if s.recv(4) != b'ACK':
                    return False
                else:
                    
                    recv_time1=time.time() 
                    print(1)
                    print(f"[HostB] 数据迁移Token时间: {recv_time1 - start_time1:.6f} 秒")    
     
                start_time2=time.time()
                # 发送KV部分
                serialized_kv = pickle.dumps(kv_data)
                header_kv = b'KVKV' + len(serialized_kv).to_bytes(4, 'big')               
                s.sendall(header_kv)

                total_sent1 = 0
                print(f"serialized_kv={len(serialized_kv)}")
                
                while total_sent1 < len(serialized_kv):
                    chunk = serialized_kv[total_sent1:total_sent1+self.chunk_size]
                    s.sendall(chunk)
                    total_sent1 += len(chunk)
                    #print(f"{total_sent1, len(serialized_kv)}")
                   # self._simulate_network_delay(len(chunk))  # 对每个块模拟延迟

                if s.recv(4) == b'ACK':    
                    start_time2= kv_data["start_time"]
                    recv_time2=time.time() 
                    print(f"[HostB] 数据迁移KV时间: {recv_time2 - start_time2:.6f} 秒")    
                    print(2)                
                recv_time3=time.time() 
                print(f"[HostB] 总数据迁移时间: {recv_time3 - start_time1:.6f} 秒")    

                return  True
        except Exception as e:
            print(f"迁移失败: {e}")
            return False

def translate_with_migration(transformer, de_sentence, client):

    # De分词
    de_tokens,de_ids=de_preprocess(de_sentence)
    if len(de_tokens)>SEQ_MAX_LEN:
        raise Exception('不支持超过{}的句子'.format(SEQ_MAX_LEN))
    
    transformer.decoder.open_kvcache() # 开启KV Cache
    enc_x_batch=torch.tensor([de_ids],dtype=torch.long).to(DEVICE)      # 准备encoder输入
    #en_token_ids=de_ids.copy()# 翻译结果
    en_token_ids=[BOS_IDX]
    print(f"enc_x_batch=={enc_x_batch} ")
    en_token_ids_total= de_ids.copy()
    start_time=time.time()


    try:
         
        while len(en_token_ids) < SEQ_MAX_LEN:

            if client.should_migrate(start_time):
                print(len(en_token_ids_total))
                total_len = len(en_token_ids_total)
                split_idx = int(total_len * client.migration_ratio)
                
                # 分割Token序列
                token_part = en_token_ids_total[:split_idx]
             
               
                # 构建迁移数据包
                token_data  = {
                    'token_part': token_part,
                    'start_time': time.time(),
                    'en_token_ids':en_token_ids,#记录了新生成的TokenID
                    'enc_x_batch': enc_x_batch.to(DEVICE),#历史TokenID
                }
                #print(f"en_token_ids={len(en_token_ids)},enc_x_batch={enc_x_batch}")
                kv_data = {
                    'kv_part': {
                        'start_idx': split_idx,
                        'cache': {}},
                    'start_time': time.time()   
                        }     

                # 提取后半部分KV缓存
                for layer_idx, layer in enumerate(transformer.decoder.decoder_blocks):
                    layer_cache = {}
                    
                    # 自注意力KV截取
                    self_k = layer.self_attn.kv_cache['K'][:, split_idx:, :]
                    self_v = layer.self_attn.kv_cache['V'][:, split_idx:, :]
                    layer_cache['self_attn'] = {'K': self_k.to(DEVICE), 'V': self_v.to(DEVICE)}
 
                    kv_data['kv_part']['cache'][f'layer_{layer_idx}'] = layer_cache
                    #migration_data['kv_part']['cache'][f'layer_{layer_idx}'] = layer_cache
                    #print(f"总数: {layer.self_attn.kv_cache['K'].shape}")
    
                print(f"Token: {len(token_data['token_part'])}")
                print(f"KV:  { kv_data['kv_part']['cache'][f'layer_{layer_idx}']['self_attn']['K'].shape}")
               
               
                if client.send_migration_data(token_data, kv_data):
                    return "[迁移触发]"
                


            if  en_token_ids == [BOS_IDX]:
                prefill = True
            else:
                prefill = False       

            # 正常生成流程
            dec_x_batch = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z=transformer.decode(dec_x_batch,enc_x_batch,prefill)#,page_id)   # decoder解碼
            
           # en_token_ids.append(torch.argmax(decoder_z[0, -1]).item())
           # en_token_ids_total.append(torch.argmax(decoder_z[0, -1]).item())   

            #dec_x_batch=torch.tensor([en_token_ids],dtype=torch.long).to(DEVICE)  # 准备decoder输入
            #decoder_z=transformer.decode(dec_x_batch,enc_x_batch,prefill)#,page_id)   # decoder解碼
            next_token_probs=decoder_z[0,dec_x_batch.size(-1)-1,:]    # 序列下一个词的概率
            next_token_id=torch.argmax(next_token_probs)    # 下一个词ID
            en_token_ids.append(next_token_id)
            en_token_ids_total.append(next_token_id) 

            if next_token_id==EOS_IDX:  # 结束符
                if len(en_token_ids)>=len(de_tokens):
                    break
      


    finally:
        transformer.decoder.close_kvcache()

if __name__ == '__main__':
    start_time1=time.time()
    transformer = Transformer(
        enc_vocab_size=len(de_vocab), dec_vocab_size=len(en_vocab),
        emb_size=512, q_k_size=64, v_size=64, f_size=2048,
        head=8, nblocks=6, dropout=0.1, seq_max_len=SEQ_MAX_LEN
    ).to(DEVICE)
    transformer.load_state_dict(torch.load('checkpoints/model.pth'))
    transformer.eval()
    client = MigrationClient(('192.168.207.213', 12345))
    
    '''
    result = translate_with_migration(
        transformer, 
         'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. ',
         client
    )
    end_time1=time.time()   
    #print(f"[HostB] 数据迁移时间: {end_time1 - start_time1:.6f} 秒")
    text = 'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.'
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)
    ''' 
     
     
    result = translate_with_migration(
        transformer, 
        'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.',
         client
    )
    end_time1=time.time()   
    #print(f"[HostB] 数据迁移时间: {end_time1 - start_time1:.6f} 秒")
    text ='Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.'
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)
     

    '''
    result = translate_with_migration(
        transformer, 
        'Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.',
         client
    )
    end_time1=time.time()   
    #print(f"[HostB] 数据迁移时间: {end_time1 - start_time1:.6f} 秒")
    text ='Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht. Heute scheint die Sonne hell und warm am blauen Himmel, während der Wind sanft durch die grünen Bäume weht.'
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

    print("本地结果:", result)
    '''  

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

    '''726
    result = translate_with_migration(
        transformer, 
        'Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park. Ein Kind spielt mit einem Ball, während ein Hund daneben sitzt. Die Sonne scheint hell und die Vögel singen in den Bäumen. Eine Frau liest ein Buch auf einer Bank, während ein Mann telefoniert. Am Horizont sieht man hohe Berge und einen blauen Himmel. Ein kleines Mädchen hält einen roten Luftballon und lacht. Die Straßen sind voller Menschen, die einkaufen oder spazieren gehen. Ein alter Mann füttert Tauben auf dem Marktplatz. Ein Junge fährt mit seinem Fahrrad schnell die Straße hinunter. Neben ihm läuft sein Hund und bellt fröhlich. Ein Auto hält an der Ampel, während Fußgänger die Straße überqueren. In einem Café sitzen Freunde zusammen und trinken Kaffee. Ein Kellner bringt ihnen frische Croissants. Die Stadt ist lebendig und voller Geräusche. Eine Gruppe von Touristen macht Fotos von einem alten Gebäude. Eine Mutter schiebt einen Kinderwagen und spricht mit einer Freundin. Der Wind weht sanft durch die Blätter der Bäume. Eine Katze sitzt auf einem Fensterbrett und beobachtet die Menschen unten. Die Glocken einer Kirche läuten zur vollen Stunde. Ein Straßenmusiker spielt eine Melodie auf seiner Gitarre. Die Menschen bleiben stehen und hören zu. Ein kleines Kind klatscht begeistert in die Hände. Eine Straßenbahn fährt vorbei und bringt die Menschen zu ihren Zielen. In einem Park machen einige Leute Yoga auf einer grünen Wiese. Ein Mann liest die Zeitung, während er einen Kaffee trinkt. Ein Obdachloser sitzt mit seinem Hund an einer Straßenecke. Die Lichter der Stadt beginnen zu leuchten, während die Sonne untergeht. Ein Paar hält Händchen und genießt den Abend. Die Straßenlaternen werfen lange Schatten auf das Kopfsteinpflaster. Eine Gruppe von Jugendlichen lacht und macht Witze. Ein Künstler malt ein Bild von der Stadt auf einer Leinwand. Die Nacht bricht herein, aber die Stadt bleibt wach. Menschen tanzen in einem Club zur Musik. Ein Taxifahrer wartet auf Fahrgäste an der Straßenecke. Die Fenster der Gebäude leuchten in warmem Licht. Eine Frau schaut aus dem Fenster und denkt nach. Ein Mann joggt am Fluss entlang, während die Stadt schläft. Der Mond scheint hell am Himmel, und die Sterne funkeln. Die Geräusche der Nacht sind leiser, aber die Stadt lebt weiter. Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park. Ein Kind spielt mit einem Ball, während ein Hund daneben sitzt. Die Sonne scheint hell und die Vögel singen in den Bäumen. Eine Frau liest ein Buch auf einer Bank, während ein Mann telefoniert. Am Horizont sieht man hohe Berge und einen blauen Himmel. Ein kleines Mädchen hält einen roten Luftballon und lacht. Die Straßen sind voller Menschen, die einkaufen oder spazieren gehen. Ein alter Mann füttert Tauben auf dem Marktplatz. Ein Junge fährt mit seinem Fahrrad schnell die Straße hinunter. Neben ihm läuft sein Hund und bellt fröhlich. Ein Auto hält an der Ampel, während Fußgänger die Straße überqueren. In einem Café sitzen Freunde zusammen und trinken Kaffee. Ein Kellner bringt ihnen frische Croissants. Die Stadt ist lebendig und voller Geräusche. Eine Gruppe von Touristen macht Fotos von einem alten Gebäude. Eine Mutter schiebt einen Kinderwagen und spricht mit einer Freundin. Der Wind weht sanft durch die Blätter der Bäume. Eine Katze sitzt auf einem Fensterbrett und beobachtet die Menschen unten. Die Glocken einer Kirche läuten zur vollen Stunde. Ein Straßenmusiker spielt eine Melodie auf seiner Gitarre. Die Menschen bleiben stehen und hören zu. Ein kleines Kind klatscht begeistert in die Hände. Eine Straßenbahn fährt vorbei und bringt die Menschen zu ihren Zielen. In einem Park machen einige Leute Yoga auf einer grünen Wiese. Ein Mann liest die Zeitung, während er einen Kaffee trinkt. Ein Obdachloser sitzt mit seinem Hund an einer Straßenecke. Die Lichter der Stadt beginnen zu leuchten, während die Sonne untergeht. Ein Paar hält Händchen und genießt den Abend. Die Straßenlaternen werfen lange Schatten auf das Kopfsteinpflaster. Eine Gruppe von Jugendlichen lacht und macht Witze. Ein Künstler malt ein Bild von der Stadt auf einer Leinwand. Die Nacht bricht herein, aber die Stadt bleibt wach. Menschen tanzen in einem Club zur Musik. Ein Taxifahrer wartet auf Fahrgäste an der Straßenecke. Die Fenster der Gebäude leuchten in warmem Licht. Eine Frau schaut aus dem Fenster und denkt nach. Ein Mann joggt am Fluss entlang, während die Stadt schläft. Der Mond scheint hell am Himmel, und die Sterne funkeln. Die Geräusche der Nacht sind leiser, aber die Stadt lebt weiter.',
         client
    )
      
    text ='Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park. Ein Kind spielt mit einem Ball, während ein Hund daneben sitzt. Die Sonne scheint hell und die Vögel singen in den Bäumen. Eine Frau liest ein Buch auf einer Bank, während ein Mann telefoniert. Am Horizont sieht man hohe Berge und einen blauen Himmel. Ein kleines Mädchen hält einen roten Luftballon und lacht. Die Straßen sind voller Menschen, die einkaufen oder spazieren gehen. Ein alter Mann füttert Tauben auf dem Marktplatz. Ein Junge fährt mit seinem Fahrrad schnell die Straße hinunter. Neben ihm läuft sein Hund und bellt fröhlich. Ein Auto hält an der Ampel, während Fußgänger die Straße überqueren. In einem Café sitzen Freunde zusammen und trinken Kaffee. Ein Kellner bringt ihnen frische Croissants. Die Stadt ist lebendig und voller Geräusche. Eine Gruppe von Touristen macht Fotos von einem alten Gebäude. Eine Mutter schiebt einen Kinderwagen und spricht mit einer Freundin. Der Wind weht sanft durch die Blätter der Bäume. Eine Katze sitzt auf einem Fensterbrett und beobachtet die Menschen unten. Die Glocken einer Kirche läuten zur vollen Stunde. Ein Straßenmusiker spielt eine Melodie auf seiner Gitarre. Die Menschen bleiben stehen und hören zu. Ein kleines Kind klatscht begeistert in die Hände. Eine Straßenbahn fährt vorbei und bringt die Menschen zu ihren Zielen. In einem Park machen einige Leute Yoga auf einer grünen Wiese. Ein Mann liest die Zeitung, während er einen Kaffee trinkt. Ein Obdachloser sitzt mit seinem Hund an einer Straßenecke. Die Lichter der Stadt beginnen zu leuchten, während die Sonne untergeht. Ein Paar hält Händchen und genießt den Abend. Die Straßenlaternen werfen lange Schatten auf das Kopfsteinpflaster. Eine Gruppe von Jugendlichen lacht und macht Witze. Ein Künstler malt ein Bild von der Stadt auf einer Leinwand. Die Nacht bricht herein, aber die Stadt bleibt wach. Menschen tanzen in einem Club zur Musik. Ein Taxifahrer wartet auf Fahrgäste an der Straßenecke. Die Fenster der Gebäude leuchten in warmem Licht. Eine Frau schaut aus dem Fenster und denkt nach. Ein Mann joggt am Fluss entlang, während die Stadt schläft. Der Mond scheint hell am Himmel, und die Sterne funkeln. Die Geräusche der Nacht sind leiser, aber die Stadt lebt weiter. Zwei Männer unterhalten sich mit zwei Frauen in einem großen Park. Ein Kind spielt mit einem Ball, während ein Hund daneben sitzt. Die Sonne scheint hell und die Vögel singen in den Bäumen. Eine Frau liest ein Buch auf einer Bank, während ein Mann telefoniert. Am Horizont sieht man hohe Berge und einen blauen Himmel. Ein kleines Mädchen hält einen roten Luftballon und lacht. Die Straßen sind voller Menschen, die einkaufen oder spazieren gehen. Ein alter Mann füttert Tauben auf dem Marktplatz. Ein Junge fährt mit seinem Fahrrad schnell die Straße hinunter. Neben ihm läuft sein Hund und bellt fröhlich. Ein Auto hält an der Ampel, während Fußgänger die Straße überqueren. In einem Café sitzen Freunde zusammen und trinken Kaffee. Ein Kellner bringt ihnen frische Croissants. Die Stadt ist lebendig und voller Geräusche. Eine Gruppe von Touristen macht Fotos von einem alten Gebäude. Eine Mutter schiebt einen Kinderwagen und spricht mit einer Freundin. Der Wind weht sanft durch die Blätter der Bäume. Eine Katze sitzt auf einem Fensterbrett und beobachtet die Menschen unten. Die Glocken einer Kirche läuten zur vollen Stunde. Ein Straßenmusiker spielt eine Melodie auf seiner Gitarre. Die Menschen bleiben stehen und hören zu. Ein kleines Kind klatscht begeistert in die Hände. Eine Straßenbahn fährt vorbei und bringt die Menschen zu ihren Zielen. In einem Park machen einige Leute Yoga auf einer grünen Wiese. Ein Mann liest die Zeitung, während er einen Kaffee trinkt. Ein Obdachloser sitzt mit seinem Hund an einer Straßenecke. Die Lichter der Stadt beginnen zu leuchten, während die Sonne untergeht. Ein Paar hält Händchen und genießt den Abend. Die Straßenlaternen werfen lange Schatten auf das Kopfsteinpflaster. Eine Gruppe von Jugendlichen lacht und macht Witze. Ein Künstler malt ein Bild von der Stadt auf einer Leinwand. Die Nacht bricht herein, aber die Stadt bleibt wach. Menschen tanzen in einem Club zur Musik. Ein Taxifahrer wartet auf Fahrgäste an der Straßenecke. Die Fenster der Gebäude leuchten in warmem Licht. Eine Frau schaut aus dem Fenster und denkt nach. Ein Mann joggt am Fluss entlang, während die Stadt schläft. Der Mond scheint hell am Himmel, und die Sterne funkeln. Die Geräusche der Nacht sind leiser, aber die Stadt lebt weiter.'
    sequence_length = len(text.split())  # 按空格分词，计算单词数
    print(f"输入序列的长度（单词数）: {sequence_length}")

 
    print("本地结果:", result)
    '''