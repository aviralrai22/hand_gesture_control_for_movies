import cv2
import time
import numpy as np
import mediapipe as mp
import math
import pyautogui

# --- Setup ---
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# --- Cooldown Variables ---
# We use separate cooldowns so you can do different actions quickly
last_action_time = 0 
cooldown = 1.0  # Wait 1 second before allowing another skip/pause

print("System Ready. Make sure VLC is the active window!")

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if len(lmList) != 0:
        # Common Point: Thumb Tip (4)
        x1, y1 = lmList[4][1], lmList[4][2]

        # --- 1. VOLUME CONTROL (Thumb + Index 8) ---
        # Continuous action (no cooldown needed)
        x2, y2 = lmList[8][1], lmList[8][2]
        vol_dist = math.hypot(x2 - x1, y2 - y1)
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # Magenta line for volume

        if vol_dist > 150:
            pyautogui.press('volumeup') # VLC: 'up' or 'volumeup'
        elif vol_dist < 30:
            pyautogui.press('volumedown') # VLC: 'down' or 'volumedown'

        
        # --- CHECK COOLDOWN FOR TRIGGERS ---
        # Only check triggers if enough time has passed
        if (time.time() - last_action_time) > cooldown:
            
            # --- 2. FORWARD (Thumb + Middle 12) ---
            x_mid, y_mid = lmList[12][1], lmList[12][2]
            forward_dist = math.hypot(x_mid - x1, y_mid - y1)
            
            if forward_dist < 30:
                print("Forward >>")
                pyautogui.press('right') # Standard VLC skip
                last_action_time = time.time()
                cv2.circle(img, (x_mid, y_mid), 20, (0, 255, 0), cv2.FILLED)

            # --- 3. REWIND (Thumb + Ring 16) ---
            x_ring, y_ring = lmList[16][1], lmList[16][2]
            rewind_dist = math.hypot(x_ring - x1, y_ring - y1)

            if rewind_dist < 30:
                print("<< Rewind")
                pyautogui.press('left') # Standard VLC rewind
                last_action_time = time.time()
                cv2.circle(img, (x_ring, y_ring), 20, (0, 255, 0), cv2.FILLED)

            # --- 4. PLAY/PAUSE (Thumb + Pinky 20) ---
            x_pinky, y_pinky = lmList[20][1], lmList[20][2]
            pause_dist = math.hypot(x_pinky - x1, y_pinky - y1)

            if pause_dist < 30:
                print("Play/Pause ||")
                pyautogui.press('space')
                last_action_time = time.time()
                cv2.circle(img, (x_pinky, y_pinky), 20, (0, 0, 255), cv2.FILLED)

    # Frame Rate
    cv2.putText(img, f'Controls Active', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    
    cv2.imshow("VLC Gesture Master", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break