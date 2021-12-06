<!-- ![header](https://capsule-render.vercel.app/api?type=slice&color=auto&height=200&text=HuPoE&fontAlign=70&rotate=13&fontAlignY=25&desc=Human%20Pose%20Estimation&descAlign=70&descAlignY=47) -->

# HuPoE



| **제목**   |중/소 산업현장 안전시스템 프로젝트 |
| :---: | :---: | 
| **부제**   |Human Pose Estimation Project with PyQt, Pyside|
| **개발자**   |`변의혁`, `구태완`|
| **개발기간**   |`2021.09` ~ `2021.11 `| 
| **포트폴리오 링크** | [포트폴리오 보기](https://www.miricanvas.com/v/1ojqh0) |
| **사용기술** | `Python`,`Tensorflow`,`Keras`,`Mediapipe`,`Dlib`,`PyQt5`,`Pyside2` |
| **버전** | v0.0 |

## 목차
1. [개요](#개요)
2. [내용](#내용)
3. [구성](#기능)
4. [결과](#결과)



## 개요
### 기획의도
 - 제품을 생산하는 공장 혹은 건설현장에서는 크고 작은 안전사고가 발생하고 있다. 특히 안전관리 사각지대로 꼽히고 있는 중ㆍ소형 건설현장 및 중소기업 산업현장 같은 경우 작업인원이 많지 않기 때문에 혼자 작업을 하는 일이 발생하고 그 경우 사고 발생 시 사후 조치가 힘들어 인명피해로 이어지기도 한다. 이를 방지하고 효율적인 대처를 위해 기획하였다. 영상 기반 행동 패턴 분석을 활용한 안전 시스템은 인공지능과 사물인터넷 등 4차 산업기술 사용으로 대규모의 비용이나 인력 투입 없이 사고 및 위험요소를 사전에 감지하거나 사고가 발생한 인원 발생 시 신속한 초동대처로 위험요소를 선제적으로 예방하는 데 목적이 있습니다.

### 개발일정
![20211205_014826](https://user-images.githubusercontent.com/84761763/144717566-e9187c72-b6f1-480a-933a-8295591ee489.png)

### 데이터 셋
 - [학습된 데이터 셋](https://drive.google.com/file/d/1GcxSzzDbk1N9Z6yUkTPrt-YcfjGv6lJJ/view?usp=sharing) 다운 받아 파일 내에 압축 풀어서 사용
 - models 폴더는 해당폴더에 놓고 사용, font 내 폴더는 windows/font에 복사하거나 실행하여서 사용
 - 학습결과

![20211205_182457](https://user-images.githubusercontent.com/84761763/144740993-b4b5e421-9381-4027-a6aa-afdef566c84f.png)


## 내용
### 주요 기능
 - meidapipe 이용 사람의 관절 추정
 - Dlib이용 사람얼굴 구분하여 신원특정
 - PyQt5, Pyside2 이용 로컬에서 이용가능한 프로그램 제작

### 기능 플로우
 - 데이터베이스에 사용자를 질의하여 있는지 확인하는 1차 로그인 인증을 거칩니다.
 - 1차 인증이 끝나면 얼굴 인식을 합니다.
 - mediapipe로 사람의 관절을 추출하고 그 관절의 길이와 각도를 이용하여 학습되어있는 걷기,앉기,쓰러지기를 구분하여 정상행동과 비정상행동을 구분 탐지합니다.
 - 그 후, 비정상 행동의 경우 30초, 60초 단위로 지속될 경우 화면에 경고메세지 발생과 알람발생 그리고 지정된 관리자에게로 카카오톡 메세지가 전송됩니다.

## 구성
### 화면구성
#### db.py
 - QtSql.QSqlQuery 클래스의 exec_ 메소드를 이용하여 쿼리문으로 데이터베이스에 사용자 정보 작성가능
 - 예제
    - create table userdata (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, username VARCHAR(100) NOT NULL, password VARCHAR(100) NOT NULL);
    - insert into userdata(username, password) values('admin', '1234');
#### loginApp.py, LoginUi.py
 ![main](https://user-images.githubusercontent.com/84761763/144741765-7833dc0b-bb97-4665-b899-c6ab58ed7583.png)
 - db.py에서 만든 데이터베이스를 기반으로 쿼리문을 조회하여 아이디와 비밀번호가 존재하는 것인지 확인한 후 로그인 합니다.
 - 1차인증
 - 초기 ID / PW
    - admin / 1234
#### face_r.py
 ![1](https://user-images.githubusercontent.com/84761763/144741535-4197f07f-99c0-4318-b3c3-51e03b579430.png)
 - 얼굴인식 페이지
 - Thread 클래스의 run 메소드의 cap = cv2.VideoCapture(0)를 수정하여 외부 카메라, 내부 카메라, 동영상 구분가능
 - 얼굴이 등록된 사람만 인증 가능합니다.(현재 삭제, 모두 통과가능)
 - 2차인증

#### loading_main.py, ui_splash_screen.py
 ![loading](https://user-images.githubusercontent.com/84761763/144741914-f1d4379d-279f-4f5f-9c5b-6a26e68df4a0.png)
 - 인간자세 추정 모델이 구동되기까지 시간이 오래걸리기 때문에 구성한 로딩페이지입니다.

#### testfinal_main.py
 ![파이널](https://user-images.githubusercontent.com/84761763/144741984-6a44af60-d827-40a1-ae1b-7435ad78abd5.png)
 - 최종화면입니다.
 - 기타 기능 및 키는 아직 미구현입니다.

## 결과
 - [시뮬레이션 동영상 링크](https://youtu.be/lE7QpYeMWs0) 
 - 개발 스택

 ![로컬 스택](https://user-images.githubusercontent.com/84761763/144742637-9a6d2284-e31f-4cc5-95a3-3709df1d2c3e.png)

