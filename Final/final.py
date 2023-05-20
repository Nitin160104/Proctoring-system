# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import dlib
import face_recognition
import numpy as np


def save(frame, button):
   cv2.imwrite('Candidate.jpg', frame)
   button.config(text='Saved')

# Create an instance of TKinter Window or frame
win = Tk()
detector = dlib.get_frontal_face_detector()

# Set the size of the window
win.geometry('640x630')
win.title('Setup')
# Create a Label to capture the Video frames
image_label = Label(win)
image_label.grid(row=0, column=0, padx=10, pady=10)

text_label = Label(win, text='Face the camera!')
text_label.grid(row=1, column=0)

save_button = Button(win, text='Capture', state=DISABLED)
save_button.grid(row=2, column=0, padx=10, pady=10)

e = Entry(win, width=40)
e.grid(row=3, column=0)
e.insert(0, 'Enter your name here')

def exit():
   global candidate_name 
   if e.get() == 'Enter your name here' or e.get() == 'Please enter a valid name':
      e.delete(0, END)
      e.insert(0, 'Please enter a valid name')
      return

   candidate_name = e.get()
   win.destroy()

exit_button = Button(win, text='Start the exam', command=exit)
exit_button.grid(row=4, column=0, padx=10, pady=10)

cap = cv2.VideoCapture(0)

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image   
   frame = cap.read()[1]
   gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   saved_frame = frame.copy()
   faces = detector(gray_frame, 1)
   for face in faces:
      x1 = face.left()
      y1 = face.top()
      x2 = face.right()
      y2 = face.bottom()
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
   
   if len(faces) == 0:
      text_label.config(text='Face the camera!')
      save_button.config(state=DISABLED)
   
   elif len(faces) == 1:
      text_label.config(text='Great! Now hit "Capture" to save your photo.')
      save_button.config(state=NORMAL, command=lambda: save(saved_frame, save_button))
         
   else:
      text_label.config(text='Only one person allowed!')
      save_button.config(state=DISABLED)
   
   cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   image_label.imgtk = imgtk
   image_label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   image_label.after(20, show_frames)

show_frames()
win.mainloop()

cap.release()

sample_image = face_recognition.load_image_file('Candidate.jpg')
candidate_face_encoding = face_recognition.face_encodings(sample_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    candidate_face_encoding
]

known_face_names = [
    candidate_name
]

def error():
   error_win = Tk()
   error_win.geometry('240x50')
   
   message = Label(error_win, text='WARNING! Potential cheating detected.')
   message.grid(row=0, column=0, padx=10, pady=10)

   # dismiss = Button(error_win, text='Dismiss', command=error_win.destroy())
   # dismiss.grid(row=1, column=0, padx=10, pady=10)

   error_win.mainloop()

video_capture = cv2.VideoCapture(0)
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    
   # Grab a single frame of video
   ret, frame = video_capture.read()
   font = cv2.FONT_HERSHEY_SIMPLEX
   org = (50, 50)
   text = 'Press q to quit'
   fontScale = 1
   color = (0, 0, 255)
   thickness = 2
   cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

   # Only process every other frame of video to save time
   if process_this_frame:
      # Resize frame of video to 1/4 size for faster face recognition processing
      small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

      # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
      rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

      # Find all the faces and face encodings in the current frame of video
      face_locations = face_recognition.face_locations(rgb_small_frame)
      face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

      face_names = []
      for face_encoding in face_encodings:
         # See if the face is a match for the known face(s)
         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
         name = "Unknown"

         # # If a match was found in known_face_encodings, just use the first one.
         # if True in matches:
         #     first_match_index = matches.index(True)
         #     name = known_face_names[first_match_index]

         # Or instead, use the known face with the smallest distance to the new face
         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
         best_match_index = np.argmin(face_distances)
         if matches[best_match_index]:
            name = known_face_names[best_match_index]

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
      cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

      # Draw a label with a name below the face
      cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

   # Display the resulting image
   cv2.imshow('Webcam Feed', frame)

   #if "Unknown" in face_names or len(face_locations) == 0:
      #error()   
      
   # Hit 'q' on the keyboard to quit!
   if cv2.waitKey(1) == ord('q'):
      break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()