int red = 13;
int green = 2;

void setup() {
  pinMode(red, OUTPUT);
  pinMode(green,OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char val = Serial.read();
    if (val == 'g'| val =='r') {
      digitalWrite(red, HIGH);
      digitalWrite(green, LOW);
    }
    else if (val == 'b'|val=='l') {
      digitalWrite(green, HIGH);
      digitalWrite(red, LOW);
    }
    else if (val == 's') {
      digitalWrite(green, LOW);
      digitalWrite(red, LOW);
    }
  }
}

