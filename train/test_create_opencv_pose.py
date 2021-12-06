import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['walking', 'sitting', 'reposing']
seq_length = 30
secs_for_action = 60

# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
# hands = mp_pose.(
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# writer = cv2.VideoWriter('video/input_save.mp4', fourcc, 30.0,(w, h))

created_time = int(time.time())
# os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    # ret, frame = cap.read()
    # input_save = frame.copy() # 동영상 저장
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()
        
        img = cv2.flip(img, 1)
        
        
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imwrite(f'video/{action}_start.jpg', img) 
        img = cv2.resize(img, (1400, 790))
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            poselist = list()
            if result.pose_landmarks is not None:
                poselist.append(result.pose_landmarks)
                for res in poselist:
                    joint = np.zeros((33, 4))
                    # print(res)
                    for j, lm in enumerate(result.pose_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,11,13,0,12,14,11,23,25,12,24,26], :3] # Parent joint
                    v2 = joint[[11,13,15,12,14,16,23,25,27,24,26,28], :3] # Child joint

                    v = v2 - v1 # [12, 3]
                    # Normalize v
                    # 벡터 의 길이( 크기 라고도 함)
                    # 거리를 측정하기 위해 벡터로 바꿈
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        
                    # Get angle using arcos of dot product
                    # angle = arccos(dot(A,B) / (|A|* |B|))
                    # 이 공식을 사용하여 두 벡터 사이의 가장 작은 각도를 찾을 수 있습니다. 이 각도는 0도에서 180도 사이 입니다. 0도에서 360도 사이에서 필요한 경우
                    # 일반적인 ∑index set 연산
                    # einsum 연산을 통해 행렬, 벡터의 내적(Dot products), 외적(Outer products), 전치(Transpose), 행렬곱 등을 일관성있게 표현할 수 있습니다.
                    
                    #벡터 a 와 벡터 b 를 내적(dot product)하면 [벡터 a의 크기] × [벡터 b의 크기] × [두 벡터가 이루는 각의 cos값] 이 됩니다.
                    #그런데 바로 위에서 벡터들의 크기를 모두 1로 표준화시켰으므로 두 벡터의 내적값은 곧 [두 벡터가 이루는 각의 cos값]이 됩니다.
                    #따라서 이것을 코사인 역함수인 arccos에 대입하면 두 벡터가 이루는 각이 나오게 됩니다.
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,1,0,3,4,4,3,6,7,9,10],:], 
                        v[[1,2,6,6,4,5,9,9,7,8,10,11],:])) # [12,]

                    # print(angle)
                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)
                    print(angle_label)
                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)
                    mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)
            # writer.write(img)  
            img = cv2.resize(img, (1400, 790))
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break
        
        cv2.imwrite(f'video/{action}_finsh.jpg', img)     
         
        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}_2'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}_2'), full_seq_data)
        
        #  
    break
cap.release()
# writer.release()
cv2.destroyAllWindows()