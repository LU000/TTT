'''
decoder解码器, 输出当前词序列的下一个词概率
'''
from torch import nn 
import torch 
from emb import EmbeddingWithPosition
from dataset import de_preprocess,en_preprocess,train_dataset,de_vocab,PAD_IDX,en_vocab
from decoder_block import DecoderBlock
from encoder import Encoder
from config import DEVICE

class Decoder(nn.Module):
    def __init__(self,vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout=0.1,seq_max_len=5000):
        super().__init__()
        self.emb=EmbeddingWithPosition(vocab_size,emb_size,dropout,seq_max_len)

        self.decoder_blocks=nn.ModuleList()
        for i in range(nblocks):
            self.decoder_blocks.append(DecoderBlock(emb_size,q_k_size,v_size,f_size,head))#, i))
        
        # 输出向量词概率Logits
        self.linear=nn.Linear(emb_size,vocab_size)  

    def create_attention_mask(self, x, cache_length):
        """
        x: (batch_size, current_seq_len)
        cache_length: 已缓存的序列长度
        """
        current_seq_len = x.size(1)
        total_seq_len = cache_length + current_seq_len
        #print(cache_length)
        #print(current_seq_len)
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.ones(total_seq_len, total_seq_len, device=DEVICE), 
            diagonal=1
        ).bool()
        
        # 创建填充掩码
        pad_mask = (x == PAD_IDX).unsqueeze(1).expand(-1, total_seq_len, -1)
        pad_mask = pad_mask.transpose(1, 2)  # (batch_size, total_seq_len, current_seq_len)
        
        return pad_mask | causal_mask[None, :, :]

    def open_kvcache(self):
        for block in self.decoder_blocks:
            block.open_kvcache()
            
    def close_kvcache(self):
        for block in self.decoder_blocks:
            block.close_kvcache()
        
    def forward(self, x, encoder_x, prefill=False):

        if prefill is True:#prefill阶段
            attn_mask=(encoder_x==PAD_IDX).unsqueeze(1) # pad_mask:(batch_size,1,seq_len)
            attn_mask=attn_mask.expand(encoder_x.size()[0],encoder_x.size()[1],encoder_x.size()[1]) # pad_mask:(batch_size,seq_len,seq_len)
            attn_mask=attn_mask.to(DEVICE)
            #print("prefill")
            encoder_x = self.emb(encoder_x)
            for block in self.decoder_blocks:
                x = block(encoder_x, attn_mask)
            prefill=False
        
        else:#decode阶段
            x = x[:, 1:] 
            seq_len = encoder_x.size(1) + x.size(1)
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().unsqueeze(0).expand(x.size(0), -1, -1).to(DEVICE)  
            #print(f"encoder_x shape: {encoder_x.shape}, x shape before embedding: {x.shape}")
            x = self.emb(x)
            encoder_x=self.emb(encoder_x)
            #print(f"x shape after embedding: {x.shape}, {encoder_x.shape}")
            x = torch.cat([encoder_x, x], dim=1)            
          
            #print("decode")
            for block in self.decoder_blocks:
                x = block(x, attn_mask)
       #print(f"222222{attn_mask.shape}")


        
        return self.linear(x)
    
if __name__=='__main__':
    # 取2个de句子转词ID序列，输入给encoder
    de_tokens1,de_ids1=de_preprocess(train_dataset[0][0]) 
    de_tokens2,de_ids2=de_preprocess(train_dataset[1][0]) 
    # 对应2个en句子转词ID序列，再做embedding，输入给decoder
    en_tokens1,en_ids1=en_preprocess(train_dataset[0][1]) 
    en_tokens2,en_ids2=en_preprocess(train_dataset[1][1])

    # de句子组成batch并padding对齐
    if len(de_ids1)<len(de_ids2):
        de_ids1.extend([PAD_IDX]*(len(de_ids2)-len(de_ids1)))
    elif len(de_ids1)>len(de_ids2):
        de_ids2.extend([PAD_IDX]*(len(de_ids1)-len(de_ids2)))
    
    enc_x_batch=torch.tensor([de_ids1,de_ids2],dtype=torch.long).to(DEVICE)
    print('enc_x_batch batch:', enc_x_batch.size())

    # en句子组成batch并padding对齐
    if len(en_ids1)<len(en_ids2):
        en_ids1.extend([PAD_IDX]*(len(en_ids2)-len(en_ids1)))
    elif len(en_ids1)>len(en_ids2):
        en_ids2.extend([PAD_IDX]*(len(en_ids1)-len(en_ids2)))
    
    dec_x_batch=torch.tensor([en_ids1,en_ids2],dtype=torch.long).to(DEVICE)
    print('dec_x_batch batch:', dec_x_batch.size())

    # Encoder编码,输出每个词的编码向量
    enc=Encoder(vocab_size=len(de_vocab),emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8,nblocks=3).to(DEVICE)
    enc_outputs=enc(enc_x_batch)
    print('encoder outputs:', enc_outputs.size())

    # Decoder编码,输出每个词对应下一个词的概率
    dec=Decoder(vocab_size=len(en_vocab),emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8,nblocks=3).to(DEVICE)
    enc_outputs=dec(dec_x_batch,enc_outputs,enc_x_batch)
    print('decoder outputs:', enc_outputs.size())