// int Dir1Pin_A = 2;      
// int Dir2Pin_A = 3;      
// int SpeedPin_A = 10;    

// int Dir1Pin_B = 5;      
// int Dir2Pin_B = 4;      
// int SpeedPin_B = 11; 
//================================================================//
//================================================================//
int Dir1Pin_A = 3;      
int Dir2Pin_A = 2;      
int SpeedPin_A = 10;    

int Dir1Pin_B = 4;      
int Dir2Pin_B = 5;      
int SpeedPin_B = 11; 
//================================================================//
/* 
Dir1Pin_A => 자동차 기준 오른편
Dir1Pin_B => 자동차 기준 왼편
*/

//================================================================//
void setup() {
  pinMode(Dir1Pin_A, OUTPUT);             // 제어 1번핀 출력모드 설정
  pinMode(Dir2Pin_A, OUTPUT);             // 제어 2번핀 출력모드 설정
  pinMode(SpeedPin_A, OUTPUT);            // PWM제어핀 출력모드 설정

  pinMode(Dir1Pin_B, OUTPUT);             // 제어 1번핀 출력모드 설정
  pinMode(Dir2Pin_B, OUTPUT);             // 제어 2번핀 출력모드 설정
  pinMode(SpeedPin_B, OUTPUT);            // PWM제어핀 출력모드 설정

  Serial.begin(9600);
}
 
void loop() {
  String front;
  String back;

 if (Serial.available() > 0){                  
    char command = Serial.read();
    
    //===============================================================================//
    //String 문자열 나누기//
    //===============================================================================//
    // String command = Serial.readStringUntil('\n');
    // command.trim(); // 문자열의 앞뒤의 공백 제거
    // //공백을 기준으로 문자열 분리하기
    // int spaceIndex = command.indexOf(' '); // 공백의 인덱스 찾기
    // if (spaceIndex != -1) { // 공백이 존재할 경우
    //   front = command.substring(0, spaceIndex); // 'hello' 부분 추출
    //   back = command.substring(spaceIndex + 1); // 'me' 부분 추출
    // }
    // Serial.println("받은데이터 : " + command + "앞부분 : " + front + "뒷부분 : " + back);
    //===============================================================================//
    
    Serial.print("Recived command : ");
    if(command == 'g'){                    
      goForward();
      Serial.println("직진"); 
    }
    else if(command == 'b'){               
      goBack();
      Serial.println("후진"); 
    }
    else if(command == 'l'){
      goLeft();
      Serial.println("좌회전");
    }
    else if(command == 'r'){
      goRight();
      Serial.println("우회전");
    }
    else if(command == 's'){
      engineBreak();
      Serial.println("정지");
    }
    else{
      Serial.println("Wrong command");      
    }
  } 
  
}

void engineBreak(){
  digitalWrite(Dir1Pin_A, LOW);         
  digitalWrite(Dir2Pin_A, LOW);
  digitalWrite(Dir1Pin_B, LOW);       
  digitalWrite(Dir2Pin_B, LOW);
}

//전진
void goForward(){
  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  analogWrite(SpeedPin_A, 100);
  
  digitalWrite(Dir1Pin_B, HIGH);         
  digitalWrite(Dir2Pin_B, LOW);
  analogWrite(SpeedPin_B, 100);
}

//후진
void goBack(){
  digitalWrite(Dir1Pin_A, LOW);          
  digitalWrite(Dir2Pin_A, HIGH);
  analogWrite(SpeedPin_A, 100);          
  
  digitalWrite(Dir1Pin_B, LOW);          
  digitalWrite(Dir2Pin_B, HIGH);
  analogWrite(SpeedPin_B, 100);
}

//좌회전
void goLeft(){
  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  analogWrite(SpeedPin_A, 150);
  
  digitalWrite(Dir1Pin_B, LOW);         
  digitalWrite(Dir2Pin_B, HIGH);
  analogWrite(SpeedPin_B, 100);
}

//우회전
void goRight(){
  digitalWrite(Dir1Pin_A, LOW);         
  digitalWrite(Dir2Pin_A, HIGH);
  analogWrite(SpeedPin_A, 100);
  
  digitalWrite(Dir1Pin_B, HIGH);         
  digitalWrite(Dir2Pin_B, LOW);
  analogWrite(SpeedPin_B, 150);
}