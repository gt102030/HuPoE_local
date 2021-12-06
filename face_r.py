# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '3_face_R.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2, time, dlib
# import mediapipe as mp
# from tensorflow.keras.models import load_model
import requests, json
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from loading_main import SplashScreen 

# 얼굴인식 시작
class Thread(QThread):
    imageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        #self.cap = cv2.VideoCapture('D:/Human_Pose_Estimation/0test2.mp4') # 0 웹캠
        self.cap = cv2.VideoCapture(0) # 0 웹캠
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
        self.descs = np.load('img/descs.npy',allow_pickle=True)[()] # ,allow_pickle=True
        while self.ThreadActive :
            ret, img = self.cap.read()
            #img = cv2.flip(img, 1)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)      
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = self.detector(img, 1)
            for k, d in enumerate(dets):
                shape = self.sp(img, d)
                face_descriptor = self.facerec.compute_face_descriptor(img, shape)

                last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

                for name, saved_desc in self.descs.items():
                    dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

                    if dist < last_found['dist']:
                        last_found = {'name': name, 'dist': dist, 'color': (0,255,0)}

            #수정부분 시작    
                    cv2.rectangle(img, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
                    cv2.putText(img, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)
                    h, w, ch = img.shape
                    bytesPerLine = ch * w
                    imge = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    resizedImg = imge.scaled(300, 300, Qt.KeepAspectRatio)
                    self.imageUpdate.emit(resizedImg)
            #끝




# 스레드 끝


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(853, 578)
        # 추가
        Form.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        Form.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # 추가끝
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(60, 30, 361, 420))
        self.widget.setStyleSheet("QPushButton#pushButton{\n"
        "    background-color:rgba(85,98,112,255);\n"
        "    color:rgba(255,255,255,200);\n"
        "    border-radius:5px;\n"
        "}\n"
        "QPushButton#pushButton:pressed{\n"
        "    padding-left:5px;\n"
        "    padding-top:5px;\n"
        "    background-color:rgba(255,107,107,255);\n"
        "    background-pasition:calc(100% - 10px)center;\n"
        "}\n"
        "QPushButton#pushButton:hover{\n"
        "    background-color:rgba(255,107,107,255);\n"
        "}")
        self.widget.setObjectName("widget")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(40, 25, 270, 360))
        self.label_2.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(85, 98, 112, 255), stop:1 rgba(255, 107, 107, 255));\n"
        "border-radius:10px;")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(70, 60, 171, 31))
        font = QtGui.QFont()
        font.setFamily("타이포_쌍문동 스텐실")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color:rgba(0, 0, 0, 200);\n"
        "")
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setGeometry(QtCore.QRect(80, 310, 191, 40))
        # 버튼이벤트 추가
        self.pushButton.clicked.connect(self.thirdWIndow)
        # 버튼 끝
        font = QtGui.QFont()
        font.setFamily("타이포_쌍문동 스텐실")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(60, 50, 181, 41))
        font = QtGui.QFont()
        font.setFamily("타이포_쌍문동 스텐실")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color:rgba(255, 255, 255, 220);\n"
        "")
        self.label_4.setObjectName("label_4")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(75, 101, 201, 191))
        self.label.setStyleSheet("border-radius:10px;")
        self.label.setObjectName("label")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.Form = Form

        #쑤래드시작
        self.Thread = Thread()
        self.Thread.start()
        self.Thread.imageUpdate.connect(self.imageUpdateslot)
        
        #끝
        

    #쓰레드 업데이트 함수
    def imageUpdateslot(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))





        


    
    # 테스트 시작 쓰레드
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
    
    # 테스트 끝 쓰레드





        # 버튼 시작
        
    
    def thirdWIndow(self):
        print('a')
        a = True
        if a: # 여기다가 얼굴 인증 조건 추가
            win = QtWidgets.QWidget()
            QtWidgets.QMessageBox.about(win, "Recognition", "사용자 인증 완료")
            #self.Form.hide()
            self.third_WIndow()
        else : 
            win = QtWidgets.QWidget()
            QtWidgets.QMessageBox.about(win, "Recognition", "사용자 인증 실패")
            exit()

    def third_WIndow(self):
        print('b')
        #Form = QtWidgets.QWidget()
        #ui = Ui_Form()
        #ui.setupUi(Form)
        self.ui = SplashScreen()
        self.ui.show()
        self.Form.close()

        # 함수끝




    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_3.setText(_translate("Form", "HuPoE System"))
        self.pushButton.setText(_translate("Form", "L o g   I n"))
        self.label_4.setText(_translate("Form", "HuPoE System"))
        self.label.setText(_translate("Form", " ")) # 동영상 나오는 프레임



#running = True

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())



