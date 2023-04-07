import serial
import time

ser = serial.Serial('COM4', 9600)

print("아두이노아 통신 시작")

while True:
    value = input("보낼값 : ");
    ser.write(value.encode());
    time.sleep(2)
    #line = ser.readline().decode("utf-8")
    #print(line)