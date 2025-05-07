import cv2
import os

# Initialize the face cascade classifier
casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(casc_path)
cam = cv2.VideoCapture(0)


# Create a folder to store saved images
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

# Counter to save images with unique filenames
counter = 0

while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with improved parameters
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Display text near the face
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check if 's' key is pressed to save face
        if cv2.waitKey(1) & 0xFF == ord("s"):
            # Save the detected face as a new image
            face_img = frame[y:y + h, x:x + w]
            filename = f"saved_faces/face_{counter}.jpg"
            cv2.imwrite(filename, face_img)
            print(f"Face saved as {filename}")
            counter += 1

    # Display the resulting frame
    cv2.imshow("True - AI", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()