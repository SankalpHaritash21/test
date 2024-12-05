import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
try:
    emotion_model = load_model('emotion_recognition_model_improved.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize DNN Face Detector
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',  # Prototxt file for the model architecture
    'res10_300x300_ssd_iter_140000_fp16.caffemodel'  # Pretrained Caffe model
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            break

        # Prepare the image for the DNN detector
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Detect faces
        face_net.setInput(blob)
        detections = face_net.forward()

        # Process each detection
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Filter detections by confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype('int')

                # Extract and preprocess face ROI
                roi_gray = frame[y:y2, x:x2]
                roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0  # Normalize pixel values
                roi_gray = np.expand_dims(roi_gray, axis=[0, -1])  # Add batch and channel dimensions

                # Predict emotion
                predictions = emotion_model.predict(roi_gray, verbose=0)
                max_index = np.argmax(predictions)
                emotion = emotion_labels[max_index]
                emotion_confidence = predictions[0][max_index] * 100

                # Draw rectangle and emotion label
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion} ({emotion_confidence:.1f}%)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                            (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Real-Time Emotion Detection', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
