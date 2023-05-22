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
int MaxSpeed = 95;
int MinSpeed = 70;

int basicSpeed = 85;
int CurrentSpeed = 0;
int LeftSpeed = 0;
int RightSpeed = 0;

bool goFlag = true;
bool leftFlag = true;
bool rightFlag = true;


String head;
String eye;
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
  Serial.println("Start");
}
 
void loop() {
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
    //   head = command.substring(0, spaceIndex); // 'hello' 부분 추출
    //   eye = command.substring(spaceIndex + 1); // 'me' 부분 추출
    // }
    // Serial.println("받은데이터 : " + command + " 앞부분 :" + head + " 뒷부분 :" + eye);
    //===============================================================================//

    // String command = Serial.readString();
    Serial.print("Recived command : ");
    Serial.print(command);
    //headCommand
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
//후진
void goBack() {
  digitalWrite(Dir1Pin_A, LOW);          
  digitalWrite(Dir2Pin_A, HIGH);  
  analogWrite(SpeedPin_A, 95);      

  digitalWrite(Dir1Pin_B, LOW);          
  digitalWrite(Dir2Pin_B, HIGH);
  analogWrite(SpeedPin_B, 95);
}

//정지
void engineBreak(){
  leftFlag = true;
  rightFlag = true;
  goFlag = true;
  if(CurrentSpeed != 0){
    CurrentSpeed -= 20;
    if(CurrentSpeed <= 0){
      CurrentSpeed = 0;
      Serial.print("완전정지상태");
      Serial.print(CurrentSpeed);
      digitalWrite(Dir1Pin_A, LOW);         
      digitalWrite(Dir2Pin_A, LOW);
      digitalWrite(Dir1Pin_A, LOW);         
      digitalWrite(Dir2Pin_A, LOW);
      goFlag = true;
    }
  }
  if(CurrentSpeed <= 0){
    Serial.print("완전정지상태");
    Serial.print(CurrentSpeed);
    digitalWrite(Dir1Pin_A, LOW);         
    digitalWrite(Dir2Pin_A, LOW);
    digitalWrite(Dir1Pin_A, LOW);         
    digitalWrite(Dir2Pin_A, LOW);
    goFlag = true;
  }
  Serial.print("정지입니다 : ");
  Serial.print(CurrentSpeed);
  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  analogWrite(SpeedPin_A, CurrentSpeed);
  
  digitalWrite(Dir1Pin_B, HIGH);         
  digitalWrite(Dir2Pin_B, LOW);
  analogWrite(SpeedPin_B, CurrentSpeed);
}

//전진
void goForward(){
  initCurrentSpeed();

  Serial.print("직진입니다 : ");
  Serial.print(CurrentSpeed);

  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  analogWrite(SpeedPin_A, CurrentSpeed);
  
  digitalWrite(Dir1Pin_B, HIGH);         
  digitalWrite(Dir2Pin_B, LOW);
  analogWrite(SpeedPin_B, CurrentSpeed);
}

//좌회전
void goLeft(){
  initLeftRightSpeed(0);

  rightFlag = true;

  // RightSpeed += 1;
  // LeftSpeed -= 10;

  // RightSpeed = 170;
  // LeftSpeed = 60;

  if(LeftSpeed <= 0){
      LeftSpeed = 0;
  }
  // if(RightSpeed > MaxSpeed){
  //   RightSpeed = MaxSpeed;
  //   Serial.print("좌회전 최고속도 도달");
  // }

  Serial.print("R : ");
  Serial.print(RightSpeed);
  Serial.print("  L : ");
  Serial.print(LeftSpeed);

  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  analogWrite(SpeedPin_A, RightSpeed);
  
  digitalWrite(Dir1Pin_B, HIGH);         
  digitalWrite(Dir2Pin_B, LOW);
  analogWrite(SpeedPin_B, LeftSpeed);
}

//우회전
void goRight(){
  initLeftRightSpeed(1);

  leftFlag = true;

  // RightSpeed -= 10;
  // LeftSpeed += 1;

  // RightSpeed = 45;
  // LeftSpeed = 95;


  if(RightSpeed <= 0){
      RightSpeed = 0;
  }
  // if(LeftSpeed > MaxSpeed){
  //   LeftSpeed = MaxSpeed;
  //   Serial.print("우회전 최고속도 도달");
  // }

  Serial.print("R : ");
  Serial.print(RightSpeed);
  Serial.print("  L : ");
  Serial.print(LeftSpeed);

  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  analogWrite(SpeedPin_A, RightSpeed);
  
  digitalWrite(Dir1Pin_B, HIGH);         
  digitalWrite(Dir2Pin_B, LOW);
  analogWrite(SpeedPin_B, LeftSpeed);
}
//==========================================================================//
//==========================================================================//
void initCurrentSpeed(){
  if(goFlag){
    Serial.print("현재속도 초기화하러 왔습니다");
    // CurrentSpeed = basicSpeed;
    CurrentSpeed = 155;
    // CurrentSpeed = 95;
    goFlag = false;
  }
  else{
    CurrentSpeed = 85;
  }
}

void initLeftRightSpeed(int control){
  // if(CurrentSpeed == 0){
    // CurrentSpeed = basicSpeed;
  // }
  //좌회전 호출됬을때
  if(leftFlag && control == 0){
    Serial.print("좌회전호출로초기화");
    // RightSpeed = CurrentSpeed;
    // LeftSpeed = CurrentSpeed - 10;
    // RightSpeed = basicSpeed;
    // LeftSpeed = basicSpeed;
    RightSpeed = 140;
    LeftSpeed = 35;
    leftFlag = false;
  }
  //우회전 호출됬을때
  else if(rightFlag && control == 1){
    Serial.print("우회전호출로초기화");
    // RightSpeed = CurrentSpeed - 10;
    // LeftSpeed = CurrentSpeed;
    // RightSpeed = basicSpeed;
    // LeftSpeed = basicSpeed;
    RightSpeed = 35;
    LeftSpeed = 140;
    rightFlag = false;
  }
  // if(testFlag && control == 2) {
  //   Serial.print("좌우속도 초기화");
  //   LeftSpeed = basicSpeed;
  //   RightSpeed = basicSpeed;
  //   testFlag = false;
  // }
}

void saveCurrentSpeed(bool boolean){
  Serial.print("현재속도 저장하러 왔습니다");
  if(boolean){
    CurrentSpeed = RightSpeed;
  }
  else{
    CurrentSpeed = LeftSpeed;
  }
  Serial.print("저장했습니다");
  Serial.print(CurrentSpeed);
}

//==========================================================================//
//==========================================================================//