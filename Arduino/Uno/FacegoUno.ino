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
  Serial.println("Start");
}
 
void loop() {
 if (Serial.available()){                  
    char command = Serial.read();          
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