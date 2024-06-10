import cv2
import pickle
import numpy as np
import face_recognition
import requests
import time
import schedule
import threading
recordedIds = []
data = {"data": recordedIds}

# The API endpoint
url = "http://localhost:4000/getTodaysPresent"

# Define the job to send recorded IDs
def job():
    # Send a POST request to the API
    response = requests.post(url, json=data)  # Use json parameter to send JSON data
    print("Recorded IDs sent: ", recordedIds)

    # Print the response
    response_json = response.json()
    print("Response from server: ", response_json)

# Function to run the scheduled jobs
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Schedule the job to run every day at 13:45
schedule.every().day.at("15:27").do(job)

# Start the scheduling thread
schedule_thread = threading.Thread(target=run_schedule)
schedule_thread.start()

cap = cv2.VideoCapture(1)  # Use 0 for default webcam
cap.set(3, 640)
cap.set(4, 480)

# Load encoded images and IDs
with open("EncodedFile.p", 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown, studentIds = encodeListKnownWithIds

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            if studentIds[matchIndex] not in recordedIds:
                cv2.putText(img, "Attendance Marked!!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                recordedIds.append(studentIds[matchIndex])
                print(recordedIds)

    cv2.imshow("webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
