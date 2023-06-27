import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import socket

actions = ['right', 'left']
seq_length = 30

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5053)

model = load_model('models/model.h5')

# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []

# bboxInfo = {}


while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()
    Pose_lmList = []
    Right_Hand_lmList = []
    Pose_data = []
    Right_Hand_data = []

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.pose_landmarks is not None:
        res = result.pose_landmarks
        pose_joint = np.zeros((33, 4))
        Right_hand_joint = np.zeros((21, 4))

        # Pose

        for p_j, lm in enumerate(result.pose_landmarks.landmark):
            pose_joint[p_j] = [lm.x, lm.y, lm.z, lm.visibility]
            h, w, c = img.shape
            cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
            Pose_lmList.append([p_j, cx, cy, cz])

        for lm in Pose_lmList:
            # lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'
            Pose_data.extend([lm[1], img.shape[0] - lm[2], lm[3], ])
            # print(data)
        # posList.append(lmString)
        # print(data)
        sock.sendto(str.encode("PoseData:" + str(Pose_data)), serverAddressPort)

        # Right Hand
        if result.right_hand_landmarks is not None:
            for h_j, h_lm in enumerate(result.right_hand_landmarks.landmark):
                Right_hand_joint[h_j] = [h_lm.x, h_lm.y, h_lm.z, h_lm.visibility]
                h, w, c = img.shape
                px, py, pz = int(h_lm.x * w), int(h_lm.y * h), int(h_lm.z * w)
                Right_Hand_lmList.append([h_j, px, py, pz])

            for lm in Right_Hand_lmList:
                # lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'
                Right_Hand_data.extend([lm[1], img.shape[0] - lm[2], lm[3], ])
                # print(data)
            # posList.append(lmString)
            # print(data)
            # sum_data = [Pose_data, Right_Hand_data]
            sock.sendto(str.encode("RightHandData:" + str(Right_Hand_data)), serverAddressPort)

        # Compute pose_angles between joints
        # Compute pose_angles between joints
        pose_v1 = pose_joint[
                  [11, 13, 15, 15, 15, 17, 11, 23, 25, 27, 27, 29, 12, 14, 16, 16, 16, 18, 12, 24, 26, 28, 28, 30],
                  :3]  # Parent joint
        pose_v2 = pose_joint[
                  [13, 15, 21, 17, 19, 19, 23, 25, 27, 29, 31, 31, 14, 16, 22, 18, 20, 20, 24, 26, 28, 30, 32, 32],
                  :3]  # Child pose_joint
        pose_v = pose_v2 - pose_v1
        # Normalize v
        pose_v = pose_v / np.linalg.norm(pose_v, axis=1)[:, np.newaxis]

        # Get pose_angle using arcos of dot product
        # pose_angle = np.arccos(np.einsum('nt,nt->n',
        #     v[[0, 1, 1, 1, 0, 6, 7, 8, 8, 12, 13, 13, 12, 18, 20, 20], :],
        #     v[[1, 2, 4, 3, 6, 7, 8, 9, 10, 13, 15, 16, 18, 19, 23, 21], :]))

        pose_angle = np.arccos(np.einsum('nt,nt->n',
                                         pose_v[[0, 12], :],
                                         pose_v[[1, 13], :]))

        pose_angle = np.degrees(pose_angle)  # Convert radian to degree

        d = np.concatenate([pose_joint.flatten(), pose_angle])

        seq.append(d)

        # mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

        # # Draw face landmarks
        mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        # # Right hand
        mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # # Left Hand
        mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.95:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

        cv2.putText(img, f'{this_action.upper()}',
                    org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break