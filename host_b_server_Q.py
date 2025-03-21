import socket
import pickle
import threading
import torch
import math
import torch.nn.functional as F
import time

HOST = '0.0.0.0'  # 监听所有网络接口
PORT = 12345      # 监听端口

# 全局 KV 存储
kv_store = {}
store_lock = threading.Lock()  # 用于线程安全的锁

def recv_all(conn, size):
    """确保完整接收 size 字节的数据"""
    received_data = b''  # 用于存储接收的数据
    while len(received_data) < size:
        chunk = conn.recv(min(4096, size - len(received_data)))  # 每次最多接收 4096 字节
        if not chunk:
            raise ConnectionError("Connection closed unexpectedly.")
        received_data += chunk  # 将接收到的块添加到接收数据中
    return received_data

def attention_calculation(kv_cache, Q, head=8, q_k_size=64, v_size=64):
    """执行多头注意力计算"""
    try:
         
        K = torch.tensor(kv_cache.get('K', []))  # 键向量
        V = torch.tensor(kv_cache.get('V', []))  # 值向量
       # print(f"[DEBUG] Q原始形状: {Q.shape} | 总元素数: {Q.numel()}")
       # print(f"[DEBUG] K原始形状: {K.shape} | 总元素数: {K.numel()}")
       # print(f"[DEBUG] V原始形状: {V.shape} | 总元素数: {V.numel()}")
        if Q.shape[0] == 0 or K.shape[0] == 0 or V.shape[0] == 0:
            return "Invalid KV cache data"

        batch_size, seq_len, hidden_dim = Q.shape

        # **1️⃣ 重新整理形状 (batch_size, seq_len, hidden_dim) → (batch_size, head, seq_len, head_dim)**
        q=Q.view(Q.size()[0],Q.size()[1],head,q_k_size).transpose(1,2) # q: (batch_size,head,seq_len,q_k_size)
        k=K.view(K.size()[0],K.size()[1],head,q_k_size).transpose(1,2).transpose(2,3) # k:(batch_size,head,q_k_size,seq_len)
        attn = torch.matmul(q, k) / math.sqrt(q_k_size)  # 计算注意力权重
         
             # 注意力分值处理
        # attn_mask: (batch_size,seq_len,seq_len)
        #attn_mask=attn_mask.unsqueeze(1).expand(-1,head,-1,-1) # attn_mask: (batch_size,head,seq_len,seq_len)
        attn_mask = kv_cache.get('attn_mask', None)
        if attn_mask is not None:
            attn_mask = torch.tensor(attn_mask).unsqueeze(1).expand(-1, head, -1, -1)
           # print(f"[DEBUG] attn_mask shape (after expand): {attn_mask.shape}")
       # else:
            #print("[DEBUG] attn_mask is None, skipping mask application")
        if attn_mask is not None:
            attn=attn.masked_fill(attn_mask,-1e9)
        attn=torch.softmax(attn,dim=-1) # scores: (batch_size,head,seq_len,seq_len)
        
        v=V.view(V.size()[0],V.size()[1],head, v_size).transpose(1,2) # v: (batch_size,head,seq_len,v_size)
        z=torch.matmul(attn,v) # z: (batch_size,head,seq_len,v_size)
        z=z.transpose(1,2) # z: (batch_size,seq_len,head,v_size)
       #print(z.size())
        z=z.reshape(z.size()[0],z.size()[1],-1)   
        return z.tolist()  # 转换为可序列化格式
    except Exception as e:
        return f"Attention calculation error: {str(e)}"

def handle_client_connection(conn, addr):
    """处理客户端连接（存储/检索 KV 数据）"""
    #print(f"[+] Connected by {addr}")
    conn.settimeout(10)  # 设置超时为10秒

    try:
        # 接收消息头（操作类型 + 数据大小）
        header = conn.recv(9)  # 操作类型（5字节） + 数据大小（4字节）
        if not header:
            print(f"[-] No header received from {addr}. Closing connection.")
            return

        # 解析操作类型和数据大小
        operation = header[:5].decode('utf-8').strip()  # 操作类型（STORE 或 RETRIEVE）
        size = int.from_bytes(header[5:], byteorder='big')  # 数据大小（4字节）
        #print(f"[*] Received operation from {addr}: {operation}, expecting {size} bytes of data.")

        if operation == "STORE":
            # 接收数据
            received_data = recv_all(conn, size)

            # 尝试解析数据并存储到 KV 存储
            try:
                kv_data = pickle.loads(received_data)  # 尝试解包并存储为键值数据
                with store_lock:
                    kv_store.update(kv_data)  # 使用线程锁保证线程安全
              #  print(f"[✔] Stored KV data from {addr}: {kv_data:[100]}")
            except pickle.UnpicklingError as e:
              #  print(f"[X] Error unpickling data from {addr}: {e}. Storing raw data.")
                with store_lock:
                    kv_store["RAW_DATA"] = received_data  # 如果解析失败，则存储为原始数据

        elif operation == "RETRI":
            # 接收 Q 向量的大小
            q_size_data = conn.recv(4)
            if not q_size_data:
               # print("[X] Failed to receive Q size.")
                return

            q_size = int.from_bytes(q_size_data, byteorder='big')
           # print(f"[*] Receiving Q vector, size: {q_size} bytes")

            received_q_data = recv_all(conn, q_size)
            Q = pickle.loads(received_q_data)  # 解析 Q

            with store_lock:
                #print('time:')
                start_time = time.time()
                attn_result = attention_calculation(kv_store, torch.tensor(Q))  # 计算注意力结果
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f'{elapsed_time:.6f}')
            serialized_data = pickle.dumps(attn_result)  # 序列化计算结果
            size = len(serialized_data)

            # 发送数据大小
            conn.sendall(size.to_bytes(4, byteorder='big'))

            # 发送计算结果
            conn.sendall(serialized_data)
           # print(f"[✔] Sent attention result to {addr}")
        else:
            print(f"[-] Unknown operation received from {addr}: {operation}")

    except ConnectionError as e:
        print(f"[X] Connection error with {addr}: {e}")
    except Exception as e:
        print(f"[X] Error handling connection from {addr}: {e}")
    finally:
        conn.close()  # 关闭连接
        #print(f"[+] Connection with {addr} closed.")

def start_server():
    """启动服务器"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()  # 开始监听端口
        #print(f"[Server] Started on {HOST}:{PORT}. Waiting for connections...")

        while True:
            conn, addr = server_socket.accept()  # 接受新的连接
            threading.Thread(target=handle_client_connection, args=(conn, addr), daemon=True).start()  # 每个客户端连接启动一个新线程

if __name__ == '__main__':
    start_server()
