import cv2
import mediapipe as mp
import numpy as np
import time, os
import math

actions = ['right', 'left']
seq_length = 30
secs_for_action = 30  # 알맞게 조절

# MediaPipe hands model
# mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# angle = []

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks is not None:
                res = result.pose_landmarks
                joint = np.zeros((33, 4))

                # nan_count = sum(math.isnan(x) for x in angle)
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[
                         [11, 13, 15, 15, 15, 17, 11, 23, 25, 27, 27, 29, 12, 14, 16, 16, 16, 18, 12, 24, 26, 28, 28,
                          30], :3]  # Parent joint
                    v2 = joint[
                         [13, 15, 21, 17, 19, 19, 23, 25, 27, 29, 31, 31, 14, 16, 22, 18, 20, 20, 24, 26, 28, 30, 32,
                          32], :3]  # Child joint
                    v = v2 - v1
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    # angle = np.arccos(np.einsum('nt,nt->n',
                    #     v[[0, 1, 1, 1, 0, 6, 7, 8, 8, 12, 13, 13, 12, 18, 20, 20], :],
                    #     v[[1, 2, 4, 3, 6, 7, 8, 9, 10, 13, 15, 16, 18, 19, 23, 21], :]))

                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 12], :],
                                                v[[1, 13], :]))

                    nan_count = sum(math.isnan(x) for x in angle)
                    # print(nan_count)
                    if nan_count <= 0:
                        angle = np.degrees(angle)  # Convert radian to degree
                        print(angle)

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, idx)

                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
