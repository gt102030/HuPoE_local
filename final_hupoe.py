from tensorflow.keras.models import load_model
import cv2, time, dlib
import mediapipe as mp
import requests, json
import numpy as np
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

class Thread_fin(QThread): #
    fin_imageUpdate = pyqtSignal(QImage)
    def fin_run(self, selectcamera='D:/Human_Pose_Estimation/test2.mp4'):
        self.ThreadActive = True
        cap = cv2.VideoCapture(selectcamera)
        actions = ['walking', 'sitting', 'reposing']
        seq_length = 30
        colors = [(245,117,16), (117,245,16), (16,117,245)]
        model = load_model('models/model_1.h5')
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()
        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        seq = []
        action_seq = []
        time_list = []
        reposing_warning_time = 10
        reposing_danger_time = 30
        img_warning = 'img/warning.png'
        img_danger = 'img/danger.png'
        warning = cv2.imread('img/warning.png')
        danger = cv2.imread('img/danger.png')
        warning = cv2.resize(warning,(150, 70))
        danger = cv2.resize(danger, (150, 70))
        i = 0
        # 얼굴 인식 -------------------------------------------------------
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
        koo = cv2.imread('img/koo.jpg')
        koo = cv2.resize(koo,(150, 150))
        byeon = cv2.imread('img/byeon1.jpg')
        byeon = cv2.resize(byeon,(150, 150))
        descs = np.load('img/descs.npy',allow_pickle=True)[()] # ,allow_pickle=True

#        def encode_face(img):
#            dets = detector(img, 1)
#
#            if len(dets) == 0:
#                return np.empty(0)
#
#            for k, d in enumerate(dets):
#                shape = sp(img, d)
#                face_descriptor = facerec.compute_face_descriptor(img, shape)
#
#                return np.array(face_descriptor)
        # 뼈다귀 --------------------------------------------------------------------------------
        while self.ThreadActive:#####수정#####
            ret, img = cap.read()
            img0 = img.copy()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) ########## 판단 요망########
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            poselist = list()
            if result.pose_landmarks is not None:
                poselist.append(result.pose_landmarks)
                for res in poselist:
                    joint = np.zeros((33, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    # Compute angles between joints
                    v1 = joint[[0,11,13,0,12,14,11,23,25,12,24,26], :3] # Parent joint
                    v2 = joint[[11,13,15,12,14,16,23,25,27,24,26,28], :3] # Child joint
                    v = v2 - v1 # [11, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,1,0,3,4,4,3,6,7,9,10],:], 
                        v[[1,2,6,6,4,5,9,9,7,8,10,11],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])

                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)
                    if len(seq) < seq_length:
                        continue
                    
                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                    reslt = model.predict(input_data)[0]
                    y_pred = model.predict(input_data).squeeze()

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.9:
                        continue

                    action = actions[i_pred]
                    action_seq.append(action)
                    
                    if len(action_seq) < 3:
                        continue

                    this_action = '?'
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action
                        if this_action == 'reposing':
                            time_list.append(time.time())
                            if time.time() - time_list[0] > reposing_warning_time and reposing_danger_time > time.time() - time_list[0]:
                                rows, cols, channels = warning.shape
                                roi = img[10:10+rows , 260:260+cols]
                                logoGray = cv2.cvtColor(warning,cv2.COLOR_BGR2GRAY)
                                ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)
                                im2_bg = cv2.bitwise_and(roi, roi, mask=mask)
                                im1_fg = cv2.bitwise_and(warning,warning,mask=mask)
                                dst = cv2.add(im2_bg, im1_fg)
                                img[10:10+rows , 260:260+cols] = dst
                                # 카카오톡 
                                i += 1
                                if i < 2:   
                                    with open("json/kakao_code.json","r") as fp:
                                        tokens = json.load(fp)

                                    url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

                                    # kapi.kakao.com/v2/api/talk/memo/default/send 

                                    headers={
                                        "Authorization" : "Bearer " + tokens["access_token"]
                                    }

                                    data={
                                        "template_object": json.dumps({
                                            "object_type":"text",
                                            "text":last_found['name']+"님이 누워 일어나지 않습니다 *경고*(10초)",
                                            "link":{
                                                "web_url":"www.naver.com"
                                            }
                                        })
                                    }
                                    response = requests.post(url, headers=headers, data=data)
                                    response.status_code        
                                elif int(time.time()) - int(time_list[0]) == reposing_danger_time:
                                    i = 0
                                elif time.time() - time_list[0] > reposing_danger_time:    
                                    rows, cols, channels = danger.shape
                                    roi = img[10:10+rows , 260:260+cols]
                                    logoGray = cv2.cvtColor(danger,cv2.COLOR_BGR2GRAY)
                                    ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)
                                    im2_bg = cv2.bitwise_and(roi, roi, mask=mask)
                                    im1_fg = cv2.bitwise_and(danger,danger,mask=mask)
                                    dst = cv2.addWeighted(im2_bg, 0, im1_fg, 1, 0)
                                    img[10:10+rows , 260:260+cols] = dst
                                    # 카카오 톡
                                    i += 1     
                                    if i < 2:
                                        with open("json/kakao_code.json","r") as fp:
                                            tokens = json.load(fp)

                                        url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

                                        # kapi.kakao.com/v2/api/talk/memo/default/send 

                                        headers={
                                            "Authorization" : "Bearer " + tokens["access_token"]
                                        }

                                        data={
                                            "template_object": json.dumps({
                                                "object_type":"text",
                                                "text":last_found['name']+"님이 누워 일어나지 않습니다 *위험*(60초)",
                                                "link":{
                                                    "web_url":"www.naver.com"
                                                }
                                            })
                                        }

                                        response = requests.post(url, headers=headers, data=data)
                                        response.status_code
                                    else:
                                        pass                  
                                else:
                                    pass
                            else:
                                time_list = []
                        img = img.copy()
                        for num, prob in enumerate(reslt):
                            cv2.rectangle(img, (0,345+num*40), (int(prob*100), 380+num*40), colors[num], -1)
                            cv2.putText(img, actions[num], (0, 370+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                        # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        cv2.putText(img, f'{this_action.upper()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                # 얼굴인식 -----------------------------------------------------------------------------------------------
                dets = detector(img, 1)
                for k, d in enumerate(dets):
                    shape = sp(img_rgb, d)
                    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

                    last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

                    for name, saved_desc in descs.items():
                        dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

                        if dist < last_found['dist']:
                            last_found = {'name': name, 'dist': dist, 'color': (0,255,0)}

                            if last_found['name'] == 'Koo Tae Wan':
                                rows, cols, channels = koo.shape
                                # 삽입하고자 하는 이미지의 위치의 로고크기만큼의 이미지 컷팅
                                # roi = img_bgr[10:10+rows , 260:260+cols]
                                roi = img[10:10+rows , 480:480+cols]
                                logoGray = cv2.cvtColor(koo,cv2.COLOR_BGR2GRAY)
                                ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)
                                im2_bg = cv2.bitwise_and(roi, roi, mask=mask)
                                im1_fg = cv2.bitwise_and(koo,koo,mask=mask)
                                dst = cv2.addWeighted(im2_bg, 0, im1_fg, 1, 0)
                                img[10:10+rows , 480:480+cols] = dst
                            elif last_found['name'] == 'Byeon Ui Hyeok':
                                rows, cols, channels = byeon.shape
                                roi = img[10:10+rows , 480:480+cols]
                                logoGray = cv2.cvtColor(byeon,cv2.COLOR_BGR2GRAY)
                                ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)
                                im2_bg = cv2.bitwise_and(roi, roi, mask=mask)
                                im1_fg = cv2.bitwise_and(byeon,byeon,mask=mask)
                                dst = cv2.addWeighted(im2_bg, 0, im1_fg, 1, 0)
                                img[10:10+rows , 480:480+cols] = dst
                            else:
                                pass
                        ##댑스 확인
                        cv2.rectangle(img, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
                        cv2.putText(img, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)
                        h, w, ch = img.shape
                        bytesPerLine = ch * w
                        imge = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        resizedImg = imge.scaled(300, 300, Qt.KeepAspectRatio)
                        self.fin_imageUpdate.emit(resizedImg)
#-------------------------------------------------------------
            # img = cv2.resize(img, (1400, 790))
            # cv2.imshow('img', img)
            # if cv2.waitKey(1) == ord('q'):
            #     break

# a = Thread_fin()
# a.fin_run()