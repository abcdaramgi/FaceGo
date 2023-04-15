import socket

HOST = '127.0.0.1'

PORT = 9999

#socket 객체 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#지정한 HOST와 PORT 사용하여 서버 접속
client_socket.connect((HOST, PORT))

while True:
    print("g(직진),r(우회전)은 red b,l은 green 중 입력 (정지하려면 's'입력)")
    message=input()

    #q 입력 시 종료
    if message == 'q':
        break
    #입력한 message 전송
    client_socket.sendall(message.encode())

    #메시지 수신
    data = client_socket.recv(1024)
    print('Received', repr(data.decode()))


client_socket.close()