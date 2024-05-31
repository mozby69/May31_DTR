from django.shortcuts import render
import cv2
import dlib
import os
from django.conf import settings
from django.contrib import messages
import math
import numpy as np
from django.http import JsonResponse
from django.http import StreamingHttpResponse
import face_recognition




cap = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()
width, height = 540, 380
predictor = dlib.shape_predictor("myapp/assets/pre-trained/shape_predictor_68_face_landmarks.dat")

def preprocessing(frame):
    frame = cv2.resize(frame, (224, 224))
    return frame

def calculate_eye_distance(left_eye, right_eye):
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle


def FaceCapture(request):
    return render(request, 'myapp/FaceCapture.html')

def gen_frames():
    global cap
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (width, height))
        faces = detector(frame)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def camera_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def capture_image(request):
    global cap
    try:
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (width, height))
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector(frame)

            if len(faces) == 1:
                for face in faces:
                    landmarks = predictor(gray, face)
                    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
                    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

                    angle = calculate_eye_distance(left_eye, right_eye)

                    reference_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
                    distance_between_eyes = np.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)
                    scale_factor = distance_between_eyes / reference_distance 

                    rotation_matrix = cv2.getRotationMatrix2D(left_eye, angle, scale_factor)

                    aligned_face = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    aligned_face = aligned_face[y:y + h, x:x + w]
                    

                # Ensure the snapshots directory exists
                snapshots_dir = os.path.join(settings.MEDIA_ROOT, "snapshots")
                if not os.path.exists(snapshots_dir):
                    os.makedirs(snapshots_dir)
                
                # Determine the next available filename
                existing_files = os.listdir(snapshots_dir)
                max_index = 0
                for file in existing_files:
                    if file.startswith("captured_face_") and file.endswith(".jpg"):
                        try:
                            index = int(file[len("captured_face_"):-len(".jpg")])
                            if index > max_index:
                                max_index = index
                        except ValueError:
                            continue

                next_index = max_index + 1
                filename = f"captured_face_{next_index}.jpg"
                filepath = os.path.join(snapshots_dir, filename)
                cv2.imwrite(filepath, aligned_face)

                if max_index > 0:
                    prev_index = max_index
                    if max_index > 1:
                        prev_index = max_index - 1
                    else:
                        prev_index = 1
                    previous_filename = f"captured_face_{prev_index}.jpg"
                    previous_filepath = os.path.join(snapshots_dir, previous_filename)

                    # Load previous and current images using face_recognition
                    previous_image = face_recognition.load_image_file(previous_filepath)
                    current_image = face_recognition.load_image_file(filepath)

                    # Encode the faces
                    previous_face_encodings = face_recognition.face_encodings(previous_image)
                    current_face_encodings = face_recognition.face_encodings(current_image)

                    if previous_face_encodings and current_face_encodings:
                        previous_encoding = previous_face_encodings[0]
                        current_encoding = current_face_encodings[0]

                        results = face_recognition.compare_faces([previous_encoding], current_encoding)
                        
                        if results[0]:
                                messages.success(request, 'Same Person Captured Successfully', extra_tags='import_successfully')
                                return JsonResponse({"success": True, "filename": filename, "successs":'Same Person Captured Successfully'})
                        else:
                            os.remove(filepath)
                            messages.error(request, 'Different person detected.', extra_tags='import_aborted')
                            return JsonResponse({"success": False, "error": "Different person detected."})
                else:
                    messages.success(request, 'Captured Successfully', extra_tags='import_successfully')
                    return JsonResponse({"success": True, "filename": filename,  "successs":'Captured Successfully'})
                    

            elif len(faces) > 1:
                messages.error(request, 'There are Multiple Faces Detected', extra_tags='import_aborted')
                return JsonResponse({"success": False, "error": "There are Multiple Faces Detected  "})
            else:
                messages.error(request, 'No Face Detected', extra_tags='import_aborted')
                return JsonResponse({"success": False, "error": "No face detected."})
        else:
            messages.error(request, 'Failed to read frame from camera.', extra_tags='import_aborted')
            return JsonResponse({"success": False, "error": "Failed to read frame from camera."})
    except Exception as e:
        messages.error(request, str(e), extra_tags='import_aborted')
        return JsonResponse({"success": False, "error": str(e)})

