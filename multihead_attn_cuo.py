'''
输入emb后的词序列,根据Q,K,V方法计算词与词之间的相关性,为每个词生成信息提取后的emb(与输入词1:1映射)
'''
from torch import nn 
import torch 
from dataset import de_vocab,de_preprocess,train_dataset
from emb import EmbeddingWithPosition
import math 
import uuid
import threading
# 新增 PageAttention 缓存管理器

class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size,q_k_size,v_size,head):
        super().__init__()
        self.emb_size=emb_size
        self.q_k_size=q_k_size
        self.v_size=v_size
        self.head=head

        self.w_q=nn.Linear(emb_size,head*q_k_size) # 多头
        self.w_k=nn.Linear(emb_size,head*q_k_size)
        self.w_v=nn.Linear(emb_size,head*v_size)
        
        # kvcache推理优化
        self.kv_cache={}
        self.kv_cache_type = ''
        self.current_page_id = None  # 当前处理的请求标识
        self.kv_pages = {}  # {page_id: {layer_id: {'K': tensor, 'V': tensor}}}
        self.lock = threading.RLock()  # 确保线程安全

    def create_page(self, page_id):
        """确保页面存在，如果不存在则创建"""
        with self.lock:
            if page_id not in self.kv_pages:
                self.kv_pages[page_id] = {}

    def store_kv(self, page_id, layer_id, kv_data):
        """存储指定 page_id 和 layer_id 的 K/V 数据"""
        with self.lock:
            if page_id not in self.kv_pages:
                self.create_page(page_id)  # 如果缓存中没有该 page_id，则初始化
            
            if page_id is None:
                raise ValueError("page_id 不能为 None!")
                
            self.kv_pages[page_id][layer_id] = {
                'K': kv_data['K'].detach().clone(),
                'V': kv_data['V'].detach().clone()
            }

    def get_kv(self, page_id, layer_id):
        """获取指定 page_id 和 layer_id 对应的 K/V 数据"""
        with self.lock:
            if page_id not in self.kv_pages:
                print(f"缓存未命中，正在为 page_id {page_id} 创建页面和缓存。")
                self.create_page(page_id)  # 确保页面存在
            return self.kv_pages[page_id].get(layer_id, None)
        
    def set_page_id(self, page_id):
        """设置当前请求的页面标识"""
        self.current_page_id = page_id
        if page_id not in self.kv_cache:
            # 初始化该页面的缓存条目
            self.kv_cache[page_id] = {}  # 这里可以是一个空字典或其他结构
            #print(f"Initialized cache for page_id: {page_id}")
        print(f"Set page_id: {page_id}")
    
    def auto_assign_page_id(self):
        """自动生成并分配新的页面标识"""
        # 使用uuid库生成一个新的唯一的page_id
        new_page_id = str(uuid.uuid4())  # 生成唯一的UUID
        self.set_page_id(new_page_id)  # 自动设置当前请求的页面ID
        print(f"Auto-assigned new page_id: {new_page_id}")
    
    def set_kvcache(self, kv_cache_type=''):
        """设置缓存类型并清空当前的缓存"""
        self.kv_cache_type = kv_cache_type
        self.kv_cache = {}
        #print(f"Cache cleared, type set to: {kv_cache_type}")

    def set_page_id(self, page_id):
        """设置当前页面标识"""
        self.current_page_id = page_id
        if page_id not in self.kv_pages:
            # 如果该页面缓存未初始化，则创建缓存
            self.create_page(page_id)  # 确保页面被创建
            #print(f"初始化了 page_id: {page_id} 的缓存")
       # print(f"已设置 page_id: {page_id}")

 
    def forward(self,x_q,x_k_v,attn_mask,current_page_id, layer_id):  # x_q: (batch_size,seq_len,emb_size), x_k_v:  (batch_size,seq_len',emb_size)
        
        cached_kv = self.get_kv(current_page_id, layer_id) if current_page_id else None
        #print(f"Current page_id: {current_page_id}")
        # kvcache推理加速,只有decoder推理阶段使用
        if self.kv_cache_type=='selfattn': # decoder的自注意力cache
            x_q=x_q[:,-1:,:] # (batch_size,seq_len=1,emb_size)
            x_k_v=x_k_v[:,-1:,:] # (batch_size,seq_len'=1,emb_size)
            
            q=self.w_q(x_q) # q: (batch_size,seq_len=1,head*q_k_size)
            new_k = self.w_k(x_k_v)
            new_v = self.w_v(x_k_v)
            #if 'Q' in cached_kv:
            #    q=torch.concat((self.kv_cache['Q'],q),dim=1) # 追加到前一次推理的Q末尾
            if cached_kv:  # 拼接历史缓存
                    k = torch.cat([cached_kv['K'], new_k], dim=1)
                    v = torch.cat([cached_kv['V'], new_v], dim=1)
            else:
                    k, v = new_k, new_v
                            # 更新全局缓存
            
            self.store_kv(
                    current_page_id,
                    layer_id,
                    {'K': k, 'V': v}
                )


        elif self.kv_cache_type=='crossattn': # decoder的交叉注意力cache
            x_q=x_q[:,-1:,:] # (batch_size,seq_len=1,emb_size)
            q=self.w_q(x_q) # q: (batch_size,seq_len,head*q_k_size)

            if cached_kv:  # 使用缓存的K/V
                k, v = cached_kv['K'], cached_kv['V']
            else:
                k = self.w_k(x_k_v)
                v = self.w_v(x_k_v)
                # 缓存编码器的K/V
                self.store_kv(
                        current_page_id,
                        layer_id,
                        {'K': k, 'V': v}
                )
               
        else: # 训练模式
            q=self.w_q(x_q) # q: (batch_size,seq_len,head*q_k_size)
            k=self.w_k(x_k_v) # k: (batch_size,seq_len,head*q_k_size)
            v=self.w_v(x_k_v) # v: (batch_size,seq_len,head*v_size)

        #print(f"{cached_kv}")

        # 多头兼容
        q=q.view(q.size()[0],q.size()[1],self.head,self.q_k_size).transpose(1,2) # q: (batch_size,head,seq_len,q_k_size)
        k=k.view(k.size()[0],k.size()[1],self.head,self.q_k_size).transpose(1,2).transpose(2,3) # k:(batch_size,head,q_k_size,seq_len)
        #print(f"[HostA] : K={k.shape}")
        # 注意力矩阵
        attn=torch.matmul(q,k)/math.sqrt(self.q_k_size) # (batch_size,head,seq_len,seq_len) row是q,col是k
        
        # 注意力分值处理
        # attn_mask: (batch_size,seq_len,seq_len)
        attn_mask=attn_mask.unsqueeze(1).expand(-1,self.head,-1,-1) # attn_mask: (batch_size,head,seq_len,seq_len)
        attn=attn.masked_fill(attn_mask,-1e9)
        attn=torch.softmax(attn,dim=-1) # scores: (batch_size,head,seq_len,seq_len)

        # 注意力与V相乘
        v=v.view(v.size()[0],v.size()[1],self.head,self.v_size).transpose(1,2) # v: (batch_size,head,seq_len,v_size)
        z=torch.matmul(attn,v) # z: (batch_size,head,seq_len,v_size)
        z=z.transpose(1,2) # z: (batch_size,seq_len,head,v_size)
        return z.reshape(z.size()[0],z.size()[1],-1) # z: (batch_size,seq_len,head*v_size)
    def update(self, cache_dict):#
        """更新 KV Cache"""
        self.kv_cache.update(cache_dict)

if __name__=='__main__':
    # 准备1个batch
    emb=EmbeddingWithPosition(len(de_vocab),128)
    de_tokens,de_ids=de_preprocess(train_dataset[0][0]) # 取de句子转词ID序列
    de_ids_tensor=torch.tensor(de_ids,dtype=torch.long)
    emb_result=emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    print('emb_result:', emb_result.size())

    # 多头注意力
    multihead=MultiHeadAttention(emb_size=128,q_k_size=256, v_size=512, head=8)
    attn_mask=torch.zeros((1,de_ids_tensor.size()[0],de_ids_tensor.size()[0])) # batch中每个样本对应1个注意力矩阵
    multihead_result=multihead(x_q=emb_result,x_k_v=emb_result,attn_mask=attn_mask)
    print('multihead_result:', multihead_result.size())

     
     