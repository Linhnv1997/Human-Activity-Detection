# #server.py 

# import socket 
# import time

# # Định nghĩa host và port mà server sẽ chạy và lắng nghe
# host = '210.123.42.177'
# port = 80
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((host, port))

# s.listen(1) # 1 ở đây có nghĩa chỉ chấp nhận 1 kết nối
# print("Server listening on port", port)

# c, addr = s.accept()
# print("Connect from ", str(addr))

# #server sử dụng kết nối gửi dữ liệu tới client dưới dạng binary

# # data = c.recv(1024)
# # while data:
# #     print("Receive",data.decode())
# #     data = c.recv(1024)
# while(1):
#     c.send(b"Hello, how are you")
#     # time.sleep(1)
# c.close()


# ############################
import socket

host = '210.123.42.177'
port = 80
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(2)

while True:
    client, addr = s.accept()

    try:
        print('connected by', addr)
        while True:
            data = client.recv(1024)
            str_data = data.decode("utf8")
            if str_data =="quit":
                break
            data = eval(str_data)
            print("Client: ",len(data))
            # msg = input("Server: ")
            

    finally:
        client.close()

s.close()