from imutils.video import videostream
import cv2
import glob
import face_recognition
from PIL import ImageDraw



video_capture=cv2.VideoCapture(0)
guy_face_encoding=[]
gay_face_encoding=[]
gay_images=glob.glob("C:/Users/Dell/Desktop/gaubuster/images/gay_images/*.jpg")
guy_images=glob.glob("C:/Users/Dell/Desktop/gaubuster/images/guy_images/*.jpg")

for guy_image in guy_images:
    guy_face=face_recognition.load_image_file(guy_image)
    encoding=face_recognition.face_encodings(guy_face)[0]
    guy_face_encoding.append(encoding)
for gay_image in gay_images:
    gay_face=face_recognition.load_image_file(gay_image)
    encoding=face_recognition.face_encodings(gay_face)[0]
    gay_face_encoding.append(encoding)

known_face_encoding=[
    guy_face_encoding+gay_face_encoding
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
            matches=face_recognition.compare_faces(face_encoding,known_face_encoding)
            name="unknown"
video_capture.release()