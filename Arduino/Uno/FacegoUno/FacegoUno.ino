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
int MaxSpeed = 130;
int MinSpeed = 70;

int basicSpeed = 85;
int CurrentSpeed = 0;
int LeftSpeed = 0;
int RightSpeed = 0;

bool goFlag = true;
bool controlFlag = true;
bool leftFlag = true;
bool rightFlag = true;

bool testFlag = true;

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
    // else if(command == "b CENTER"){               
    //   goBack();
    //   Serial.println("후진"); 
    // }
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
    // //eyeCommand
    // else if(command == "gd") {
    //   accelForward(CurrentSpeed);
    //   Serial.println("가속 전진");
    // }

    // else if(command == "g LEFT"){
    //   reduceForward();
    //   Serial.println("감속 전진"); 
    // }
    // else if(eye == 'c') {
    //   accelBack();
    //   Serial.println("가속 후진"); 
    // }
    // else if(eye == 'v') {
    //   reduceBack();
    //   Serial.println("감속 후진"); 
    // }
    // else{
    //   Serial.println("Wrong command");      
    // }
  } 
  
}
//후진
void goBack() {
  digitalWrite(Dir1Pin_A, LOW);          
  digitalWrite(Dir2Pin_A, HIGH);  
  analogWrite(SpeedPin_A, 250);      

  digitalWrite(Dir1Pin_B, LOW);          
  digitalWrite(Dir2Pin_B, HIGH);
  analogWrite(SpeedPin_B, 250);
}

//정지
void engineBreak(){
  controlFlag = true;
  leftFlag = true;
  rightFlag = true;
  testFlag = true;
  if(CurrentSpeed != 0){
    CurrentSpeed -= 10;
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

  CurrentSpeed += 1;

  if(CurrentSpeed > MaxSpeed){
    CurrentSpeed = MaxSpeed;
    Serial.print("최고속도 도달");
  }

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

  // RightSpeed += 2;
  // LeftSpeed -= 1;
  RightSpeed = 170;
  LeftSpeed = 60;

  if(LeftSpeed <= 0){
      LeftSpeed = 0;
  }
  if(RightSpeed > MaxSpeed){
    RightSpeed = MaxSpeed;
    Serial.print("좌회전 최고속도 도달");
  }

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
// void goLeft(){
//   initLeftRightSpeed(0);
//   rightFlag = true;
  
//   RightSpeed += 2;
//   LeftSpeed -= 10;

//   if(LeftSpeed <= 0){
//       LeftSpeed = 0;
//   }
//   Serial.print("좌회전입니다 : ");
//   Serial.print("R : ");
//   Serial.print(RightSpeed);
//   Serial.print("  L : ");
//   Serial.print(LeftSpeed);

//   digitalWrite(Dir1Pin_A, HIGH);         
//   digitalWrite(Dir2Pin_A, LOW);
//   analogWrite(SpeedPin_A, RightSpeed);
  
//   digitalWrite(Dir1Pin_B, HIGH);         
//   digitalWrite(Dir2Pin_B, LOW);
//   analogWrite(SpeedPin_B, LeftSpeed);

//   saveCurrentSpeed(true);
// }

//우회전
void goRight(){
  initLeftRightSpeed(1);

  leftFlag = true;

  // RightSpeed -= 1;
  // LeftSpeed += 1;

  RightSpeed = 60;
  LeftSpeed = 170;


  if(RightSpeed <= 0){
      RightSpeed = 0;
  }
  if(LeftSpeed > MaxSpeed){
    LeftSpeed = MaxSpeed;
    Serial.print("우회전 최고속도 도달");
  }

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
// void goRight(){
//   initLeftRightSpeed(1);
//   leftFlag = true;

//   RightSpeed -= 10;
//   LeftSpeed += 2;

//   if(RightSpeed <= 0){
//       RightSpeed = 0;
//   }
//   Serial.print("우회전입니다 : ");
//   Serial.print("R : ");
//   Serial.print(RightSpeed);
//   Serial.print("  L : ");
//   Serial.print(LeftSpeed);

//   digitalWrite(Dir1Pin_A, HIGH);         
//   digitalWrite(Dir2Pin_A, LOW);
//   analogWrite(SpeedPin_A, RightSpeed);
  
//   digitalWrite(Dir1Pin_B, HIGH);         
//   digitalWrite(Dir2Pin_B, LOW);
//   analogWrite(SpeedPin_B, LeftSpeed);

//   saveCurrentSpeed(false);
// }
//==========================================================================//
//==========================================================================//
void initCurrentSpeed(){
  if(goFlag){
    Serial.print("현재속도 초기화하러 왔습니다");
    CurrentSpeed = basicSpeed;
    goFlag = false;
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
    RightSpeed = basicSpeed;
    LeftSpeed = basicSpeed;
    leftFlag = false;
  }
  //우회전 호출됬을때
  else if(rightFlag && control == 1){
    Serial.print("우회전호출로초기화");
    // RightSpeed = CurrentSpeed - 10;
    // LeftSpeed = CurrentSpeed;
    RightSpeed = basicSpeed;
    LeftSpeed = basicSpeed;
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

//가속 전진
void accelForward(int speed) { 
  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  
  digitalWrite(Dir1Pin_B, HIGH);         
  digitalWrite(Dir2Pin_B, LOW);

  while( speed <=100 ) {
    analogWrite(SpeedPin_A, speed);
    analogWrite(SpeedPin_B, speed);
    speed += 10; 
  }    
}

//전진 감속
void reduceForward() {
  int currentSpeed = 100;
  digitalWrite(Dir1Pin_A, HIGH);         
  digitalWrite(Dir2Pin_A, LOW);
  digitalWrite(Dir1Pin_B, HIGH);       
  digitalWrite(Dir2Pin_B, LOW);

   while (currentSpeed >= 0) {
    analogWrite(SpeedPin_A, currentSpeed);
    analogWrite(SpeedPin_B, currentSpeed);
    currentSpeed -= 10; 
    delay(200);
  }
}



//가속 후진
void accelBack(){
  int currentSpeed = 10;
  digitalWrite(Dir1Pin_A, LOW);         
  digitalWrite(Dir2Pin_A, HIGH);
  
  digitalWrite(Dir1Pin_B, LOW);         
  digitalWrite(Dir2Pin_B, HIGH);

  while( currentSpeed <=250) {
    analogWrite(SpeedPin_A, currentSpeed);
    analogWrite(SpeedPin_B, currentSpeed);
    currentSpeed += 10; 
    delay(200);
  } 
}

//후진 감속
void reduceBack() {
  int currentSpeed = 250;
  digitalWrite(Dir1Pin_A, LOW);         
  digitalWrite(Dir2Pin_A, HIGH);
  digitalWrite(Dir1Pin_B, LOW);       
  digitalWrite(Dir2Pin_B, HIGH);

   while (currentSpeed >= 0) {
    analogWrite(SpeedPin_A, currentSpeed);
    analogWrite(SpeedPin_B, currentSpeed);
    currentSpeed -= 30; 
    delay(200);
  }
}

