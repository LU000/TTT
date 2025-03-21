import socket
import pickle
import threading

HOST = '0.0.0.0'
PORT = 12345
kv_store = {}
store_lock = threading.Lock()  # 用于线程安全

def receive_all(conn, size):
    """确保完整接收 size 字节数据"""
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            raise ConnectionError("Connection closed before receiving all data")
        data += packet
    return data

def handle_client_connection(conn, addr):
    print(f"[+] Connected by {addr}")
    conn.settimeout(10)  # 设置超时10秒

    try:
        header = conn.recv(9)  # 5字节操作类型 + 4字节数据大小
        if not header:
            print(f"[-] No header received from {addr}. Closing connection.")
            return

        operation = header[:5].decode('utf-8').strip()
        size = int.from_bytes(header[5:], byteorder='big')
        print(f"[*] Received operation from {addr}: {operation}, expecting {size} bytes.")

        if operation == "STORE":
            try:
                received_data = receive_all(conn, size)
            except ConnectionError as e:
                print(f"[X] Connection error while receiving data from {addr}: {e}")
                return
            except socket.timeout:
                print(f"[X] Timeout while receiving data from {addr}")
                return

            if len(received_data) != size:
                print(f"[X] Incomplete data received from {addr}. Expected {size}, got {len(received_data)}")
                return

            try:
                kv_data = pickle.loads(received_data)
                with store_lock:
                    kv_store.clear()
                    kv_store.update(kv_data)
                print(f"[DEBUG] Current KV Store: {kv_store}")  # 添加这一行
                
                conn.sendall(b"ACK_KV_RECEIVED")

            except pickle.UnpicklingError as e:
                print(f"[X] Error unpickling data from {addr}: {e}. Storing raw data.")
                with store_lock:
                    kv_store["RAW_DATA"] = received_data

        elif operation == "RETRI":
            with store_lock:
               # print(f"[DEBUG] Sending KV Store: {kv_store}")  # 添加这一行
                serialized_data = pickle.dumps(kv_store)
            size = len(serialized_data)

            conn.sendall(size.to_bytes(4, byteorder='big'))
            conn.sendall(serialized_data)
            print(f"[✔] Sent KV data to {addr}: {size}")

        else:
            print(f"[-] Unknown operation from {addr}: {operation}")

    except ConnectionError as e:
        print(f"[X] Connection error with {addr}: {e}")
    except Exception as e:
        print(f"[X] Unexpected error handling connection from {addr}: {e}")
    finally:
        conn.close()
        print(f"[+] Connection with {addr} closed.")

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"[Server] Started on {HOST}:{PORT}. Waiting for connections...")

        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=handle_client_connection, args=(conn, addr), daemon=True).start()

if __name__ == '__main__':
    start_server()
