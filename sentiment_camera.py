pip install opencv-python deepface
import cv2
from deepface import DeepFace

# Initialize webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera not accessible")
    exit()

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert to grayscale for faster face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Analyze emotion using DeepFace
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']

            # Display emotion on screen
            cv2.putText(frame, dominant_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        except Exception as e:
            print("Analysis error:", e)

    # Display the resulting frame
    cv2.imshow('Sentiment Analysis - Press Q to Quit', frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
