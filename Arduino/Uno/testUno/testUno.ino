String incommingByte = "";

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()){
    int value = Serial.parseInt();
    Serial.print("받은값 : ");
      digitalWrite(LED_BUILTIN, HIGH);
      delay(10000);
      digitalWrite(LED_BUILTIN, LOW);
      delay(1000);
  }
}
