import cv2
import mediapipe as mp
import numpy as np
import audio

template = cv2.imread('./image/hand_template.png', cv2.IMREAD_UNCHANGED)
template_h, template_w = template.shape[:2]

vidoe_playing = False
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

def is_hand_open(hand_landmarks, h, w):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_coords = (int(index_tip.x * w), int(index_tip.y * h))
    middle_coords = (int(middle_tip.x * w), int(middle_tip.y * h))

    def distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    dist_index_middle = distance(index_coords, middle_coords)
    dist_thumb_index = distance(thumb_coords, index_coords)

    umbral = 70

    return dist_index_middle > umbral and dist_thumb_index

def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    alpha_overlay = img_overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c] = (
            alpha_overlay * img_overlay[:, :, c] + 
            alpha_background * img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c]
        )
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2

    overlay_image_alpha(frame, template, (center_x, center_y))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            if (center_x < x_min < center_x + template_w and
                center_y < y_min < center_y + template_h and
                center_x < x_max < center_x + template_w and
                center_y < y_max < center_y + template_h):

                if is_hand_open(hand_landmarks, h, w):

                    audio.play_sound('./sound/Wile.mp3')

                    video = cv2.VideoCapture('./video/vid.mp4')
                    while video.isOpened():
                        ret_vid, frame_vid = video.read()
                        if not ret_vid:
                            break
                        cv2.imshow('Hand Scanner', frame_vid)
                        if cv2.waitKey(30) & 0xFF == 27:
                            break
                    video.release()
                    vidoe_playing = False


    cv2.imshow('Hand Scanner', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

