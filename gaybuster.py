from imutils.video import videostream
import cv2
import glob
import face_recognition
from PIL import ImageDraw
import numpy as np



video_capture=cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
guy_image = face_recognition.load_image_file("C:/Users/Dell/Desktop/gaubuster/images/guy_images/aayush.jpg")
guy_face_encoding = face_recognition.face_encodings(guy_image)[0]


# Load a second sample picture and learn how to recognize it.
gay_image = face_recognition.load_image_file("C:/Users/Dell/Desktop/gaubuster/images/gay_images/gay2.jpg")
gay_face_encoding = face_recognition.face_encodings(gay_image)[0]

known_face_encoding = [
    guy_face_encoding,
    gay_face_encoding
]
known_names=[
    "guy",
    "gay"
]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # capture single frame from the video
    ret,frame=video_capture.read()
    #now we resize the video to its 1/4th quater which makes facial recognition faster
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #since face_recognition use rgb format we have to convert pixel from BGR into RGB
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name="gay"
            if True in matches:
                print(matches)
                first_match_index=matches.index(True)
                name = known_names[first_match_index]
            face_names.append(name)
            
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255,0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
