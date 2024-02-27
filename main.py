import cv2
import mediapipe as mp
import RPi.GPIO as GPIO

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def is_hand_close_to_head(hand_landmarks, xmin, ymin, width, height, image_width, image_height, proximity_threshold=50):
    # Convert normalized coordinates to pixel coordinates for the face bounding box
    face_x1 = int(xmin * image_width)
    face_y1 = int(ymin * image_height)
    face_x2 = int((xmin + width) * image_width)
    face_y2 = int((ymin + height) * image_height)
    
    # Check all hand landmarks
    for landmark in hand_landmarks.landmark:
        # Convert normalized coordinates to pixel coordinates for each hand landmark
        hand_x, hand_y = int(landmark.x * image_width), int(landmark.y * image_height)
        
        # Check if the landmark is within or near the face bounding box
        if face_x1 - proximity_threshold < hand_x < face_x2 + proximity_threshold and face_y1 - proximity_threshold < hand_y < face_y2 + proximity_threshold:
            return True
    
    return False

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image and find hands and faces
        hand_results = hands.process(image)
        face_results = face_detection.process(image)

        image.flags.writeable = True  # Enable writing on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ih, iw, _ = image.shape  # get the image height and width

        # Process face detection results
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                expansion_factor = 0.15  # Expand the hitbox by 10%
                x, y, w, h = int((bboxC.xmin - expansion_factor / 2) * iw), int((bboxC.ymin - 0.25 / 2) * ih), int((bboxC.width + expansion_factor) * iw), int((bboxC.height + expansion_factor) * ih)
                # h = h*0.2
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw the face bounding box
        
        # Process hand detection results
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw the hand landmarks and connections
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check if any hand landmark is close to or inside any face bounding box
                if face_results.detections != None:
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        if is_hand_close_to_head(hand_landmarks, bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height, iw, ih):
                            print("Hand is near the face!")
                            GPIO.output(18, GPIO.HIGH)  # Set GPIO pin 18 to high (1)
                            # Optional: You can also highlight the hand or face to indicate the detection visually
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Highlight the face bounding box
                        else:
                            GPIO.output(18, GPIO.LOW)  # Set GPIO pin 18 to low (1)
                        
        # Display the image
        cv2.imshow('Hand near Head Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()