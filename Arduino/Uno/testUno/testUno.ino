String incommingByte = "";

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

void loop() {
  if (Serial.available() > 0) { // 시리얼 버퍼에 값이 있는지 확인
    String value = Serial.readString(); // 시리얼 버퍼에서 값을 읽어옴
    Serial.print("받은 값: ");
    Serial.println(value); // 받은 값을 시리얼 모니터에 출력
  }
}
