'''
decoder block支持堆叠, 每个block都输入emb序列并输出emb序列(1:1对应)
'''
from torch import nn 
import torch 
from multihead_attn import MultiHeadAttention
from emb import EmbeddingWithPosition
from dataset import de_preprocess,en_preprocess,train_dataset,de_vocab,PAD_IDX,en_vocab
from encoder import Encoder
from config import DEVICE

class DecoderBlock(nn.Module):
    def __init__(self,emb_size,q_k_size,v_size,f_size,head):#,layer_idx):
        super().__init__()
        #self.layer_idx = layer_idx
        # 自注意力机制
        self.self_attn = MultiHeadAttention(emb_size, q_k_size, v_size, head)
        self.attn_norm = nn.LayerNorm(emb_size)

        '''
        # 第1个多头注意力 --> 自注意力
        self.first_multihead_attn=MultiHeadAttention(emb_size,q_k_size,v_size,head) 
        self.z_linear1=nn.Linear(head*v_size,emb_size) 
        self.addnorm1=nn.LayerNorm(emb_size)

        # 第2个多头注意力 --> 交叉注意力
        self.second_multihead_attn=MultiHeadAttention(emb_size,q_k_size,v_size,head) 
        self.z_linear2=nn.Linear(head*v_size,emb_size) 
        self.addnorm2=nn.LayerNorm(emb_size)
        '''
        
        # feed-forward结构
        self.feedforward=nn.Sequential(
            nn.Linear(emb_size,f_size),
            nn.ReLU(),
            nn.Linear(f_size,emb_size)
        )
        self.addnorm3=nn.LayerNorm(emb_size)

    def open_kvcache(self):
        self.self_attn.set_kvcache(kv_cache_type='selfattn')
        
    def close_kvcache(self):
        self.self_attn.set_kvcache(kv_cache_type='')
        
    def forward(self, x, attn_mask):
        # 自注意力
        attn_output = self.self_attn(x, x, attn_mask)
        #print(f"{attn_output.shape},{x.shape}")
        attn_output = self.attn_norm(attn_output + x)
        
        # 前馈网络
        ffn_output = self.feedforward(attn_output)
        return self.addnorm3(ffn_output + attn_output)

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
   #print('enc_x_batch batch:', enc_x_batch.size())

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

    # 生成decoder所需的掩码
    first_attn_mask=(dec_x_batch==PAD_IDX).unsqueeze(1).expand(dec_x_batch.size()[0],dec_x_batch.size()[1],dec_x_batch.size()[1]) # 目标序列的pad掩码
    first_attn_mask=first_attn_mask|torch.triu(torch.ones(dec_x_batch.size()[1],dec_x_batch.size()[1]),diagonal=1).bool().unsqueeze(0).expand(dec_x_batch.size()[0],-1,-1).to(DEVICE) # &目标序列的向后看掩码
    print('first_attn_mask:',first_attn_mask.size())
    # 根据来源序列的pad掩码，遮盖decoder每个Q对encoder输出K的注意力
    second_attn_mask=(enc_x_batch==PAD_IDX).unsqueeze(1).expand(enc_x_batch.size()[0],dec_x_batch.size()[1],enc_x_batch.size()[1]) # (batch_size,target_len,src_len)
    print('second_attn_mask:',second_attn_mask.size())

    first_attn_mask=first_attn_mask.to(DEVICE)
    second_attn_mask=second_attn_mask.to(DEVICE)

    # Decoder输入做emb先
    emb=EmbeddingWithPosition(len(en_vocab),128).to(DEVICE)
    dec_x_emb_batch=emb(dec_x_batch)
    print('dec_x_emb_batch:',dec_x_emb_batch.size())

    # 5个Decoder block堆叠
    decoder_blocks=[]
    for i in range(5):
        decoder_blocks.append(DecoderBlock(emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8).to(DEVICE))

    for i in range(5):
        dec_x_emb_batch=decoder_blocks[i](dec_x_emb_batch,enc_outputs,first_attn_mask,second_attn_mask)
    print('decoder_outputs:',dec_x_emb_batch.size())