import cv2
import time

# Initialize the face cascade classifier
casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(casc_path)
cam = cv2.VideoCapture(0)

# Get the default frame rate (may not always be accurate)
fps = cam.get(cv2.CAP_PROP_FPS)
print(f"Expected FPS: {fps}")

# Start time for measuring FPS
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1

    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Calculate and display FPS every second
    if time.time() - start_time >= 1:
        print(f"Actual FPS: {frame_count}")
        frame_count = 0
        start_time = time.time()

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()