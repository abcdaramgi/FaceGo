import socket
# import serial
#시리얼 통신
# ser = serial.Serial("COM5", 9600)

HOST = '127.0.0.1'

PORT = 9999

#소켓 객체 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)

server_socket.bind((HOST, PORT))

#클라이언트 접속 허용
server_socket.listen()

#accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓 리턴
client_socket, addr = server_socket.accept()

#접속한 클라이언트 주소
print('Connected by', addr)

while True:
    data = client_socket.recv(1024)

    if data:
        # 수신받은 데이터 출력
        print('받은 데이터 : ', data.decode())

        client_socket.sendall(data)



    #아두이노로 전송
    # ser.write(data)

client_socket.close()
server_socket.close()

