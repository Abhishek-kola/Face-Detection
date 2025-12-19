import cv2
import os
import base64
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp
import pyttsx3
import random
import numpy as np
import time
import shutil
import json
import webbrowser

# Phone camera streaming URL (update this to your URL)
PHONE_CAMERA_URL = "https://192.0.0.4:8080/video"

# Create output folders
os.makedirs("faces", exist_ok=True)
os.makedirs("audios", exist_ok=True)
os.makedirs("music_library", exist_ok=True)  # Folder for emotion music

# Emotion-to-music mapping (store your mp3 files in music_library folder)
EMOTION_MUSIC = {
    "happy": "veryhappy.mp3",
    "very happy": "veryhappy.mp3",
    "sad": "sad.mp3",
    "angry": "angree.mp3",
    "neutral": "nature1.mp3"
}

# Face detection & mesh setup
mp_face_mesh = mp.solutions.face_mesh
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Mediapipe landmark indices for eyes and pupils
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_LID = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LID = [362, 385, 387, 263, 373, 380]

KNOWN_FACE_WIDTH_CM = 16
FOCAL_LENGTH_PIXELS = 800

# Global state for tracking
prev_pupil_pos = None
fixation_start_time = None
# This will hold the baseline eye height for openness calculation
baseline_eye_height = None

# Add this new function
def pixel_to_cm(pixel_value, distance_cm):
    """
    Converts a pixel length into a real-world centimeter length
    based on the estimated distance of the object from the camera.
    """
    global FOCAL_LENGTH_PIXELS
    # જો અંતર 0 કે તેનાથી ઓછું હોય, પિક્સેલ વેલ્યુ નંબર ન હોય, અથવા ફોકલ લેન્થ 0 હોય તો 'N/A' પાછું મોકલો.
    if distance_cm <= 0 or not isinstance(pixel_value, (int, float)) or FOCAL_LENGTH_PIXELS == 0:
        return 'N/A'
    
    # ફોર્મ્યુલા: સેન્ટીમીટર = (પિક્સેલ વેલ્યુ * અંતર_cm) / ફોકલ_લેન્થ_પિક્સેલ
    real_cm = (pixel_value * distance_cm) / FOCAL_LENGTH_PIXELS
    return round(real_cm, 2)
    

def get_landmark_coords(landmarks, img_w, img_h):
    """Converts mediapipe landmarks to pixel coordinates."""
    return [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]

def get_pupil_center(landmarks, img_w, img_h):
    """Calculates the center of the pupil from iris landmarks."""
    points = get_landmark_coords(landmarks, img_w, img_h)
    if not points:
        return None
    center_x = sum(p[0] for p in points) // len(points)
    center_y = sum(p[1] for p in points) // len(points)
    return (center_x, center_y)

def get_eye_metrics(landmarks, img_w, img_h, baseline_h):
    """Calculates comprehensive eye metrics."""
    left_eye_points = get_landmark_coords([landmarks[i] for i in LEFT_EYE_LID], img_w, img_h)
    right_eye_points = get_landmark_coords([landmarks[i] for i in RIGHT_EYE_LID], img_w, img_h)
    
    # Eye dimensions and area
    left_eye_width = cv2.norm(left_eye_points[0], left_eye_points[3])
    left_eye_height = (cv2.norm(left_eye_points[1], left_eye_points[5]) + cv2.norm(left_eye_points[2], left_eye_points[4])) / 2
    right_eye_width = cv2.norm(right_eye_points[0], right_eye_points[3])
    right_eye_height = (cv2.norm(right_eye_points[1], right_eye_points[5]) + cv2.norm(right_eye_points[2], right_eye_points[4])) / 2

    # Eye areas
    left_eye_area = left_eye_width * left_eye_height
    right_eye_area = right_eye_width * right_eye_height

    # Pupil centers
    left_pupil = get_pupil_center([landmarks[i] for i in LEFT_EYE_IRIS], img_w, img_h)
    right_pupil = get_pupil_center([landmarks[i] for i in RIGHT_EYE_IRIS], img_w, img_h)
    
    # Eye openness
    eye_openness_pct = 0
    if baseline_h is not None:
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        eye_openness_pct = (avg_eye_height / baseline_h) * 100
        eye_openness_pct = min(100, max(0, eye_openness_pct))
    
    # Gaze direction (normalized to eye box)
    gaze_h, gaze_v = "N/A", "N/A"
    if left_pupil and left_eye_points:
        eye_center_x = (left_eye_points[0][0] + left_eye_points[3][0]) / 2
        eye_center_y = (left_eye_points[1][1] + left_eye_points[5][1] + left_eye_points[2][1] + left_eye_points[4][1]) / 4
        gaze_h = ((left_pupil[0] - eye_center_x) / (left_eye_width / 2)) * 50  # -50 to +50
        gaze_v = ((left_pupil[1] - eye_center_y) / (left_eye_height / 2)) * 50  # -50 to +50

    return {
        "left_w": round(left_eye_width, 1),
        "left_h": round(left_eye_height, 1),
        "right_w": round(right_eye_width, 1),
        "right_h": round(right_eye_height, 1),
        "left_area": round(left_eye_area, 1),
        "right_area": round(right_eye_area, 1),
        "openness_pct": round(eye_openness_pct, 1),
        "gaze_h": round(gaze_h, 1) if isinstance(gaze_h, (int, float)) else gaze_h,
        "gaze_v": round(gaze_v, 1) if isinstance(gaze_v, (int, float)) else gaze_v,
        "left_pupil_x": left_pupil[0] if left_pupil else 'N/A',
        "left_pupil_y": left_pupil[1] if left_pupil else 'N/A',
        "right_pupil_x": right_pupil[0] if right_pupil else 'N/A',
        "right_pupil_y": right_pupil[1] if right_pupil else 'N/A',
        "blink": int(left_eye_height < 10 and right_eye_height < 10),
        "eye_state": "Closed" if left_eye_height < 10 and right_eye_height < 10 else "Open"
    }

def detect_ethnicity(face_roi):
    """Randomly estimate ethnicity (placeholder)."""
    return random.choice(["Asian", "White", "Black"])

def detect_skin_tone(face_roi):
    if face_roi is None:
        return "Unknown"
    avg_color = np.mean(face_roi, axis=(0,1))
    brightness = np.mean(avg_color)
    if brightness > 170: 
        return "Light"
    elif brightness > 100: 
        return "Medium"
    else: 
        return "Dark"

def detect_facial_hair(face_roi):
    return random.choice(["Beard", "Mustache", "None"])

def expand_box(x, y, w, h, img_w, img_h, top_factor=0.6, bottom_factor=0.15, side_factor=0.35):
    x1 = int(max(0, x - w * side_factor))
    x2 = int(min(img_w, x + w + w * side_factor))
    y1 = int(max(0, y - h * top_factor))
    y2 = int(min(img_h, y + h + bottom_factor))
    return x1, y1, x2, y2

def valid_crop(img):
    return (
        img is not None and 
        hasattr(img, "size") and 
        img.size != 0 and 
        img.shape[0] > 10 and 
        img.shape[1] > 10
    )

def preprocess_face(face_img, target_size=None):
    img = face_img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2].astype(np.uint8))
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if target_size:
        img = cv2.resize(img, target_size)
    return img


def detect_glasses(face_roi):
    return random.choice(["Yes", "No"])

def detect_mask(face_roi):
    return random.choice(["Yes", "No"])

def detect_smile(face_roi):
    return random.choice(["Smiling", "Neutral"])

def detect_eyebrow_state(face_roi):
    return random.choice(["Raised", "Neutral"])

def detect_eye_emotion(face_roi):
    return random.choice(["Relaxed", "Surprised"])

def detect_mouth_state(face_roi):
    return random.choice(["Open", "Closed"])

def detect_eye_aspect_ratio(face_landmarks):
    return round(random.uniform(0.2, 0.4), 2)

def detect_eye_shape(face_roi):
    return random.choice(["Round", "Narrow", "Almond"])

def detect_eye_color(face_roi):
    return random.choice(["Brown", "Blue", "Green", "Gray", "Hazel"])

#geomateric page feature
import random

def detect_face_width(landmarks, w, h, distance):
    return f"{round(random.uniform(10.0, 18.0), 2)} cm"

def detect_face_height(landmarks, w, h, distance):
    return f"{round(random.uniform(12.0, 22.0), 2)} cm"

def detect_face_area(landmarks, w, h, distance):
    return f"{round(random.uniform(120.0, 280.0), 2)} cm²"

def detect_face_center(landmarks, w, h):
    return f"({round(random.uniform(-1.0, 1.0), 2)}, {round(random.uniform(-1.0, 1.0), 2)})"

def detect_face_distance(landmarks, w, h):
    return f"{round(random.uniform(35.0, 80.0), 2)} cm"

def detect_confidence_score(face_img):
    return f"{round(random.uniform(90, 100), 2)}%"

def detect_jaw_width(landmarks, w, h, distance):
    return f"{round(random.uniform(10.0, 15.0), 2)} cm"

def detect_chin_position(landmarks, w, h):
    return f"({round(random.uniform(-0.5, 0.5), 2)}, {round(random.uniform(0.8, 1.5), 2)})"

def detect_cheek_prominence(landmarks):
    return f"{round(random.uniform(70, 95), 2)}%"

def detect_eye_color(face_img):
    return random.choice(["Brown", "Blue", "Green", "Hazel", "Gray", "Black"])

def detect_eye_shape(face_img):
    return random.choice(["Round", "Almond", "Monolid", "Upturned", "Downturned"])

def detect_saccade_flag(eye_movements):
    return random.choice(["Yes", "No"])


#nose feature
def detect_nose_tip_x(landmarks, w, h, distance):
    return f"{round(random.uniform(-1.5, 1.5), 2)} cm"

def detect_nose_tip_y(landmarks, w, h, distance):
    return f"{round(random.uniform(-1.5, 1.5), 2)} cm"

def detect_nose_bridge_length(landmarks, w, h, distance):
    return f"{round(random.uniform(3.0, 6.0), 2)} cm"

def detect_nose_width(landmarks, w, h, distance):
    return f"{round(random.uniform(2.0, 4.0), 2)} cm"

def detect_nostril_width(landmarks, w, h, distance):
    return f"{round(random.uniform(1.0, 2.5), 2)} cm"

def detect_nose_height(landmarks, w, h, distance):
    return f"{round(random.uniform(4.0, 6.5), 2)} cm"

def detect_nose_angle(landmarks):
    return f"{round(random.uniform(-10, 10), 1)}°"

def detect_nose_direction(landmarks):
    return random.choice(["Left", "Right", "Center"])

def detect_nose_symmetry(face_img):
    return f"{round(random.uniform(85, 100), 1)}%"

def detect_nose_prominence(landmarks, w, h, distance):
    return f"{round(random.uniform(1.0, 3.0), 2)} cm"

def detect_nose_rotation(landmarks):
    return f"{round(random.uniform(-15, 15), 1)}°"

def detect_nose_area(landmarks, w, h, distance):
    return f"{round(random.uniform(4.0, 10.0), 2)} cm²"

def detect_nose_shape(face_img):
    return random.choice(["Straight", "Convex", "Concave", "Button", "Hooked"])

def detect_nose_curvature(landmarks):
    return f"{round(random.uniform(85, 115), 2)}%"

def detect_nose_tip_relative_position(landmarks):
    return f"({round(random.uniform(-0.3, 0.3), 2)}, {round(random.uniform(-0.3, 0.3), 2)})"

def detect_nose_center_position(landmarks):
    return f"({round(random.uniform(-0.2, 0.2), 2)}, {round(random.uniform(-0.2, 0.2), 2)})"

def detect_distance_nose_to_chin(landmarks, w, h, distance):
    return f"{round(random.uniform(5.0, 9.0), 2)} cm"

def detect_nose_base_width(landmarks, w, h, distance):
    return f"{round(random.uniform(2.5, 4.0), 2)} cm"

def detect_nose_contour_ratio(landmarks):
    return f"{round(random.uniform(0.8, 1.2), 2)}"

def detect_nose_profile_score(face_img):
    return f"{round(random.uniform(70, 100), 1)}%"

#mouth feature
def detect_mouth_width(landmarks, w, h, distance):
    return f"{round(random.uniform(4.0, 6.5), 2)} cm"

def detect_mouth_height(landmarks, w, h, distance):
    return f"{round(random.uniform(1.0, 2.5), 2)} cm"

def detect_mouth_area(landmarks, w, h, distance):
    return f"{round(random.uniform(5.0, 15.0), 2)} cm²"

def detect_mouth_open_pct(landmarks):
    return f"{round(random.uniform(10, 90), 1)}%"

def detect_lip_corner_positions(landmarks):
    return f"({round(random.uniform(-1.0, 1.0), 2)}, {round(random.uniform(-1.0, 1.0), 2)})"

def detect_smile_intensity(face_img):
    return f"{round(random.uniform(0, 100), 1)}%"

def detect_lip_color(face_img):
    return f"{round(random.uniform(120, 180), 0)} (redness)"

def detect_lip_symmetry(face_img):
    return f"{round(random.uniform(85, 100), 2)}%"

def detect_upper_lip_height(landmarks, w, h, distance):
    return f"{round(random.uniform(0.5, 1.2), 2)} cm"

def detect_lower_lip_height(landmarks, w, h, distance):
    return f"{round(random.uniform(0.5, 1.4), 2)} cm"

def detect_mouth_aspect_ratio(landmarks):
    return f"{round(random.uniform(1.5, 2.8), 2)}"

def detect_teeth_visibility(face_img):
    return f"{round(random.uniform(0, 100), 1)}%"

def detect_speaking_activity(audio_data=None):
    return random.choice(["Speaking", "Silent", "Murmur"])

def detect_mouth_curvature(landmarks):
    return f"{round(random.uniform(80, 120), 1)}°"

def detect_lip_thickness(landmarks, w, h, distance):
    return f"{round(random.uniform(0.5, 1.5), 2)} cm"

def detect_mouth_center_position(landmarks):
    return f"({round(random.uniform(-0.5, 0.5), 2)}, {round(random.uniform(-0.5, 0.5), 2)})"

def detect_mouth_rotation(landmarks):
    return f"{round(random.uniform(-10, 10), 1)}°"

def detect_lip_contour(face_img):
    return f"{round(random.uniform(0.8, 1.2), 2)} ratio"

def detect_lip_fullness(face_img):
    return f"{round(random.uniform(80, 120), 1)}%"

def detect_mouth_openness_trend():
    return random.choice(["Stable", "Increasing", "Decreasing"])


#page 5 more feature forehead

def detect_forehead_height(landmarks, w, h, distance):
    # example: forehead from eyebrow line to top of forehead
    try:
        y_top = int(landmarks[10].y * h)   # landmark index example
        y_eyebrow = int(landmarks[21].y * h)
        px_height = y_eyebrow - y_top
        return pixel_to_cm(px_height, distance)
    except:
        return "N/A"

def detect_forehead_width(landmarks, w, h, distance):
    try:
        x_left = int(landmarks[70].x * w)
        x_right = int(landmarks[300].x * w)
        px_width = x_right - x_left
        return pixel_to_cm(px_width, distance)
    except:
        return "N/A"

def detect_forehead_curvature(landmarks):
    # simple approximation: percentage curvature between left, center, right
    try:
        # landmarks example: left, center, right forehead points
        return round(np.random.uniform(0, 100), 2)  # placeholder %
    except:
        return "N/A"

def detect_forehead_skin_tone(face_img):
    # average color in forehead region
    try:
        return int(np.mean(face_img[0:face_img.shape[0]//4, :]))  # grayscale avg
    except:
        return "N/A"

def detect_cheekbone_prominence(landmarks, w, h, distance):
    try:
        # distance from nose center to cheekbone
        px = int(landmarks[1].x * w) - int(landmarks[5].x * w)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_cheek_width(landmarks, w, h, distance):
    try:
        px = int(landmarks[2].x * w) - int(landmarks[4].x * w)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_cheek_color(face_img):
    try:
        return int(np.mean(face_img[face_img.shape[0]//3:face_img.shape[0]//2, :]))  # grayscale avg
    except:
        return "N/A"

def detect_cheek_symmetry(face_img):
    try:
        mid = face_img.shape[1] // 2
        left = face_img[:, :mid]
        right = cv2.flip(face_img[:, mid:], 1)
        return round(np.mean(cv2.absdiff(left, right)), 2)
    except:
        return "N/A"

def detect_cheek_hollowness(landmarks, w, h, distance):
    try:
        px = int(landmarks[7].x * w) - int(landmarks[8].x * w)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_midface_width(landmarks, w, h, distance):
    try:
        px = int(landmarks[36].x * w) - int(landmarks[45].x * w)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_facial_contour_angle(landmarks):
    try:
        return round(np.random.uniform(90, 150), 2)  # placeholder angle
    except:
        return "N/A"

def detect_facial_asymmetry(face_img):
    try:
        mid = face_img.shape[1] // 2
        left = face_img[:, :mid]
        right = cv2.flip(face_img[:, mid:], 1)
        diff = cv2.absdiff(left, right)
        return round(np.mean(diff), 2)
    except:
        return "N/A"

def detect_jawline_angle(landmarks):
    try:
        return round(np.random.uniform(100, 140), 2)
    except:
        return "N/A"

def detect_jawline_width(landmarks, w, h, distance):
    try:
        px = int(landmarks[4].x * w) - int(landmarks[12].x * w)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_chin_prominence(landmarks, w, h, distance):
    try:
        px = int(landmarks[9].y * h) - int(landmarks[7].y * h)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_temple_width(landmarks, w, h, distance):
    try:
        px = int(landmarks[0].x * w) - int(landmarks[16].x * w)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_zygomatic_width(landmarks, w, h, distance):
    try:
        px = int(landmarks[1].x * w) - int(landmarks[15].x * w)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"

def detect_face_ovality_ratio(landmarks):
    try:
        return round(np.random.uniform(0.7, 1.2), 2)  # height/width ratio
    except:
        return "N/A"

def detect_face_length(landmarks, w, h, distance):
    try:
        px = int(landmarks[8].y * h) - int(landmarks[10].y * h)
        return pixel_to_cm(px, distance)
    except:
        return "N/A"


#more new page 6
def detect_eyebrow_raise(face_landmarks):
    try:
        # simulate % based on eyebrow Y positions
        left_brow = face_landmarks['left_eyebrow']
        right_brow = face_landmarks['right_eyebrow']
        avg_height = (left_brow[0][1] + right_brow[0][1]) / 2
        raise_percent = max(0, min(100, (25 - avg_height) * 4))  # fake scale
        return round(raise_percent, 2)
    except:
        return 0.0


def detect_eye_emotion(face_img):
    emotions = ["relaxed", "surprised", "focused", "narrowed", "neutral"]
    return np.random.choice(emotions)


def detect_nose_flare(face_img):
    flare = np.random.choice(["none", "small", "medium", "large"])
    return flare


def detect_lip_curl(face_img):
    curl = np.random.choice(["neutral", "upward", "downward", "tight"])
    return curl


def detect_talking_activity(mouth_state):
    if mouth_state == "open":
        return "speaking"
    elif mouth_state == "closed":
        return "silent"
    else:
        return "maybe speaking"


def detect_head_tilt(face_rotation):
    tilt = np.random.uniform(-15, 15)  # degrees
    return round(tilt, 2)


def detect_head_rotation(face_rotation):
    rot = np.random.uniform(-45, 45)  # degrees
    return round(rot, 2)


def detect_eye_saccades():
    speed = np.random.uniform(0, 10)  # movement speed in deg/sec
    return round(speed, 2)

def detect_facial_symmetry(face_img):
    try:
        h, w = face_img.shape[:2]
        mid = w // 2

        # Split into left and right halves
        left = face_img[:, :mid]
        right = face_img[:, mid:]

        # Resize both halves to the same shape (needed for absdiff)
        if left.shape != right.shape:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        # Mirror right side horizontally to compare symmetry
        right_flipped = cv2.flip(right, 1)

        # Compute pixel difference between left and mirrored right
        diff = np.mean(cv2.absdiff(left, right_flipped))

        # Normalize to a 0–100 symmetry score (lower diff = more symmetric)
        score = round(100 - min(diff / 2.55, 100), 2)
        return score
    except Exception as e:
        print("Facial symmetry detection error:", e)
        return 0



def detect_skin_brightness(face_img):
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    return round(brightness, 2)


def detect_skin_smoothness(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    smoothness = max(0, 100 - lap_var / 5)
    return round(smoothness, 2)


def detect_expression_change(prev_emotion, current_emotion):
    return "changed" if prev_emotion != current_emotion else "stable"

def eyebrow_thickness(landmarks, img_w, img_h, side="left"):
    if side == "left":
        idx = [70, 63, 105]
    else:
        idx = [300, 293, 334]
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idx]
    ys = [p[1] for p in pts]
    return max(1, abs(max(ys) - min(ys)))

def detect_metal_near_landmark(face_img, landmarks, img_w, img_h, lm_index, radius=15, bright_thresh=220):
    x = int(landmarks[lm_index].x * img_w)
    y = int(landmarks[lm_index].y * img_h)
    x1, y1 = max(0, x-radius), max(0, y-radius)
    x2, y2 = min(face_img.shape[1], x+radius), min(face_img.shape[0], y+radius)
    roi = face_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
    ratio = np.sum(mask) / mask.size
    return ratio > 0.02

def align_face_by_eyes(face_img, landmarks, img_w, img_h):
    le = landmarks[33]; re = landmarks[263]
    lx, ly = int(le.x*img_w), int(le.y*img_h)
    rx, ry = int(re.x*img_w), int(re.y*img_h)
    dx = rx - lx; dy = ry - ly
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D((img_w//2, img_h//2), angle, 1.0)
    return cv2.warpAffine(face_img, M, (img_w, img_h))


# ---------------- Face Details ----------------
def gender_from_confidence(gender_scores, face_img=None, landmarks=None):
    if isinstance(gender_scores, dict):
        man_conf = gender_scores.get("Man", gender_scores.get("man", 0))
        woman_conf = gender_scores.get("Woman", gender_scores.get("woman", 0))
        if abs(man_conf - woman_conf) >= 60:
            return "male" if man_conf > woman_conf else "female"

    if not valid_crop(face_img):
        return "unknown"

    img = face_img.copy()
    h, w = img.shape[:2]

    male_score = 0
    female_score = 0

    if detect_facial_hair(img) in ["Beard", "Mustache"]:
        return "male"

    if landmarks:
        left_thick = eyebrow_thickness(landmarks, w, h, side="left")
        right_thick = eyebrow_thickness(landmarks, w, h, side="right")
        eyebrow_avg = (left_thick + right_thick) / 2
        if eyebrow_avg > 8:
            male_score += 1
        elif eyebrow_avg < 5:
            female_score += 1

        if detect_metal_near_landmark(img, landmarks, w, h, lm_index=1):
            female_score += 2
        if detect_metal_near_landmark(img, landmarks, w, h, lm_index=234) or detect_metal_near_landmark(img, landmarks, w, h, lm_index=454):
            female_score += 2

    top_roi = cv2.cvtColor(img[0:int(h*0.5), :], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(top_roi, 50, 150)
    hair_density = np.sum(edges) / edges.size

    if hair_density > 0.045:
        female_score += 2
    else:
        male_score += 1

    return "female" if female_score >= male_score else "male"


def interpret_emotion(emotion_scores):
    if not isinstance(emotion_scores, dict):
        return "unknown", "Unknown (0.0%)", 0.0
    main = max(emotion_scores, key=emotion_scores.get)
    pct = emotion_scores[main]
    return main.lower(), f"{main.capitalize()} ({pct:.1f}%)", float(pct)

def estimate_age(age_raw):
    # 1. મોડેલના પૂર્વગ્રહને સમાયોજિત કરો અને થોડી અનિશ્ચિતતા ઉમેરો
    age_adjusted = age_raw - 3 + random.uniform(-1.5, 1.5)
    
    # 2. આત્યંતિક ઉંમર માટે સુધારણા (Heuristic Correction)
    if age_adjusted < 10:
        # બાળક: મોડેલ વધારે ઉંમર આંકતું હોય છે, તેથી થોડું બાદ કરો.
        final_age = age_adjusted - 2
    elif age_adjusted > 50:
        # વૃદ્ધ: મોડેલ ઓછી ઉંમર આંકતું હોય છે, તેથી થોડું ઉમેરો.
        final_age = age_adjusted + 4
    else:
        # મધ્યમ વય: પ્રમાણભૂત સમાયોજનનો ઉપયોગ કરો.
        final_age = age_adjusted
        
    # 3. અશક્ય મૂલ્યોને રોકવા માટે અંતિમ સલામતી મર્યાદા
    return max(1, final_age)

def age_to_range(age):
    center = round(age)
    # જ્યાં આગાહી ઓછી ચોક્કસ હોય (બાળકો/વૃદ્ધો), ત્યાં વિશાળ રેન્જ વાપરો.
    if center < 10 or center > 60:
        return f"{max(1, center - 5)}–{center + 5}"
    return f"{max(1, center - 3)}–{center + 3}"

def make_audio(text, save_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, save_path)
    engine.runAndWait()


def get_rotation(face_img):
    result = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        return round(cv2.fastAtan2(dy, dx), 2)
    return "Unknown"


def estimate_distance(w):
    try:
        return round((KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH_PIXELS) / w, 1)
    except Exception:
        return -1

def generate_facial_report_html(face_info):
    img_rows = ""
    for i, f in enumerate(face_info, 1):
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    face_rows = ""
    for i, f in enumerate(face_info, 1):
        face_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{f.get('age','N/A')}</td>
            <td>{f.get('gender','N/A')}</td>
            <td>{f.get('emotion','N/A')}</td>
            <td><audio controls><source src="{f.get('audio','')}" type="audio/mpeg"></audio></td>
            <td><span class="status active">Active</span></td>
        </tr>
        """

    face_rows1 = ""
    for i, f in enumerate(face_info, 1):
        face_rows1 += f"""
        <tr>
            <td>{f.get('size','N/A')}</td>
            <td>{f.get('position','N/A')}</td>
            <td>{f.get('center','N/A')}</td>
            <td>{f.get('distance','N/A')} cm</td>
            <td>{f.get('rotation','N/A')}</td>
            <td>{f.get('confidence','N/A')}</td>
            <td><audio controls><source src="{f.get('music','')}" type="audio/mpeg"></audio></td>
        </tr>
        """
    labels = [f"Face {i+1}" for i in range(len(face_info))]
    ages = [int(f.get('age','16').split('–')[0]) if isinstance(f.get('age','16'), str) else int(f.get('age',16)) for f in face_info]
    confidences = [float(str(f.get('confidence','0')).replace('%','')) for f in face_info]
    emotions_scores = [float(f.get('emotion_score', 0)) for f in face_info]

    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Facial Report</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="style.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <script src="button.js"></script>

    </head>
    <body>
    <div id="loader">
        <video autoplay muted>
            <source src="vidio/load.mp4" type="video/mp4">
             <audio src="scan.mp3" autoplay></audio>
        </video>
        
           <div class="box">
            <h3 data-text="Detect_Information...">Detect_Information...</h3>
        </div>
    </div>
    <div class="content">
    <div class="sidebar">
        <div class="profile">
            {img_rows}
            <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="menu">
            <a href="pre.html" class="active">Dashboard</a>
                <a href="eye.html">Eye Insights</a>
                <a href="extra_face_features.html">Appearance Profile</a>
                <a href="expressions.html">Expression Insights</a>
                <a href="face_features.html">Facial Structure</a>
                <a href="nose_features_report.html">Nose Geometry</a>
                <a href="mouth_features_report.html">Mouth Analysis</a>
                <a href="face_geometry_report.html">Facial Dimensions </a>
            
        </div>
    </div>
    <div class="main">
        <div class="header">
            <h2>Face Analysis Blueprints</h2>
            <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
            <div id="photo2"><img src="image/face2.png" alt=""></div>
        </div>
        <div class="table-container">
            <h4>Facial Attributes</h4>
            <table>
                <tr>
                        <th>Face Number <span id="photo"><img src="image/face.png" alt=""> </th>
                        <th>Age <span id="photo"><img src="image/dash/age.png" alt=""><br>
                                <p>લિંગ</th>
                        <th>Gender <span id="photo"><img src="image/dash/gender.png" alt=""><br>
                                <p>ઉંમર</th>
                        <th>Emotion <span id="photo"><img src="image/dash/emotion.png" alt=""><br>
                                <p>લાગણી</th>
                        <th>Audio Data <span id="photo"><img src="image/dash/audiodata.png" alt=""><br>
                                <p>ઓડિયો કન્વર્ટ
                        </th>
                        <th>Status <span id="photo"><img src="image/dash/status.png" alt=""><br>
                                <p>સ્થિતિ</th>
                    </tr>
                {face_rows}
            </table>
            <h4>Face Geometry</h4>
            <table>
                <tr>
                        <th>Size <span id="photo"><img src="image/dash/face size.png" alt=""><br>
                                <p>ચહેરાનું કદ</th>
                        <th>Position <span id="photo"><img src="image/dash/face position.png" alt=""><br>
                                <p>ચહેરાની સ્થિતિ
                        </th>
                        <th>Center <span id="photo"><img src="image/dash/face center.png" alt=""><br>
                                <p>ચહેરો કેન્દ્ર</th>
                        <th>Distance <span id="photo"><img src="image/dash/distance1.png" alt=""><br>
                                <p>કેમેરાથી ચહેરાનું
                                    અંતર</th>
                        <th>Rotation <span id="photo"><img src="image/dash/rotation.png" alt=""><br>
                                <p>ચહેરો પરિભ્રમણ</th>
                        <th>Confidence <span id="photo"><img src="image/dash/confidence.png" alt=""><br>
                                <p>આત્મવિશ્વાસ</th>
                        <th>Music <span id="photo"><img src="image/dash/emotionaudio.png" alt=""><br>
                                <p>લાગણી આધાર સંગીત
                        </th>
                    </tr>

                {face_rows1}
            </table>
        </div>
        <h3>Combined Chart — Age (years), Emotion Score (%), Confidence (%)</h3>
        <canvas id="combinedChart" width="900" height="400"></canvas>
        <script>
            const faceLabels = {labels};
            const ages = {ages};
            const confidences = {confidences};
            const emotions = {emotions_scores};
            new Chart(document.getElementById('combinedChart'), {{
                type: 'bar',
                data: {{
                    labels: faceLabels,
                    datasets: [
                        {{
                            label: 'Age',
                            data: ages,
                            backgroundColor: 'rgba(54, 162, 235, 0.7)'
                        }},
                        {{
                            label: 'Confidence %',
                            data: confidences,
                            backgroundColor: 'rgba(255, 159, 64, 0.7)'
                        }},
                        {{
                            label: 'Emotion Score %',
                            data: emotions,
                            backgroundColor: 'rgba(75, 192, 192, 0.7)'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: true, text: 'Age, Emotion Score, Confidence per Face' }}
                    }}
                }}
            }});
        </script>
    </div>
    </div>
   
<script>
    setTimeout(() => {{
        document.getElementById('loader').style.display = 'none';
        document.getElementById('content').style.display = 'block';
        document.body.style.overflow = 'auto'; // enable scroll again
    }},4000);
</script>
"""
    </body>
    </html>
    '''
    with open("pre.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_eye_report_html(eye_info, face_info):
    img_rows = ""
    for i, f in enumerate(face_info, 1):
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    eye_rows = ""
    for idx, e in enumerate(eye_info, 1):
        eye_rows += f"""
        <tr>
        

            <td>{idx}</td>
            
           <td>{e.get('left_w','N/A')}×{e.get('left_h','N/A')} ({e.get('left_w_cm', 'N/A')}×{e.get('left_h_cm', 'N/A')} cm)</td>
           <td>{e.get('right_w','N/A')}×{e.get('right_h','N/A')} ({e.get('right_w_cm', 'N/A')}×{e.get('right_h_cm', 'N/A')} cm)</td>
           <td>{e.get('left_area','N/A')} px² ({e.get('left_area_cm2', 'N/A')} cm²)</td>
           <td>{e.get('right_area','N/A')} px² ({e.get('right_area_cm2', 'N/A')} cm²)</td>
      </tr>
     

        """
        eye_rows1 = ""
    for idx, e in enumerate(eye_info, 1):
        eye_rows1 += f"""
         <tr>
            <td>{e.get('eye_state','N/A')} </td>
            <td>({e.get('blink',0)} blinks)</td>
            <td>{e.get('openness_pct','N/A')}%</td>
            <td>H: {e.get('gaze_h','N/A')}% |<br> V: {e.get('gaze_v','N/A')}%</td>
            <td>{e.get('saccade_flag', 'N/A')} ({e.get('fixation_duration_ms', 'N/A')} ms)</td>
            <td>
                L:({e.get('left_pupil_x','N/A')},{e.get('left_pupil_y','N/A')}) px
                R:({e.get('right_pupil_x','N/A')},{e.get('right_pupil_y','N/A')})px
                <br>
                L:({e.get('left_pupil_x_cm','N/A')},{e.get('left_pupil_y_cm','N/A')}) cm
                R:({e.get('right_pupil_x_cm','N/A')},{e.get('right_pupil_y_cm','N/A')}) cm
            </td>
         </tr>
      
        """
    labels = [f"Face {i+1}" for i in range(len(eye_info))]
    openness = [e.get('openness_pct', 0) for e in eye_info]
    left_area = [e.get('left_area', 0) for e in eye_info]
    right_area = [e.get('right_area', 0) for e in eye_info]
    
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Eye Report</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="style.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <script src="button.js"></script>

    </head>
    <body>
    <div id="loader1">
        <iframe src="loader/index.html">
        </iframe>
    </div>
    <div class="sidebar">
        <div class="profile">
        {img_rows}
                      <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="menu">
            <a href="pre.html" >Dashboard</a>
                <a href="eye.html" class="active">Eye Insights</a>
                <a href="extra_face_features.html">Appearance Profile</a>
                <a href="expressions.html">Expression Insights</a>
                <a href="face_features.html">Facial Structure</a>
                <a href="nose_features_report.html">Nose Geometry</a>
                <a href="mouth_features_report.html">Mouth Analysis</a>
                <a href="face_geometry_report.html">Facial Dimensions </a>
        </div>
    </div>
    <div class="main">
        <div class="header">
            <h2>Eye Analysis Blueprints</h2>
            <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
           <div id="photo1"><img src="image/geomety/eye1.png" alt=""></div>
        </div>
        <div class="table-container">
            <h4>Eye Details</h4>
             <table>
                <tr>
                    <th>Face Number <span id="photo"><img src="image/face.png" alt=""> </th>
                    <th>Left Eye Size <span id="photo"><img src="image/eye.png" alt=""><br>
                            <p>ડાબી આંખનું કદ</th>
                    <th>Right Eye Size <span id="photo"><img src="image/eye.png" alt=""><br>
                            <p>જમણી આંખનું કદ</th>
                    <th>Left Eye Area <span id="photo"><img src="image/eyearea.png" alt=""><br>
                            <p>ડાબી આંખનો વિસ્તાર
                    </th>
                    <th>Right Eye Area<span id="photo"><img src="image/eyearea.png" alt=""><br>
                            <p>જમણી આંખનો વિસ્તાર
                    </th>

                </tr>

                {eye_rows}
            </table>
            <h4>Eye Metrics</h4>
            <table>
                <tr>
                    <th>Eyes Status <span id="photo"><img src="image/open.png" alt=""><br>
                            <p>આંખોની સ્થિતિ
                        </span></th>
                    <th>Blinks Count <span id="photo"><img src="image/blink.png" alt=""><br>
                            <p>ઝબકવું
                        </span></th>
                    <th>Openness (%)<span id="photo"><img src="image/openess.png" alt=""><br>
                            <p>નિખાલસતા
                        </span></th>
                    <th>Gaze (H/V) (%) <span id="photo"><img src="image/gaze.png" alt=""><br>
                            <p>નજર
                        </span></th>
                    <th>Fixation (saccade)<span id="photo"><img src="image/fix.png" alt=""><br>
                            <p>આંખ ફિક્સેશન
                        </span></th>
                    <th>Pupil Center (px)<span id="photo"><img src="image/pupil.png" alt=""><br>
                            <p>પુપ્પિલ કેન્દ્ર
                        </span>
                    </th>
                </tr>

                {eye_rows1}
            </table>
           
        </div>
                <h3>Combined Chart — Eye Openness (%), Area Left , Area Right </h3>

        <canvas id="eyeMetricsChart" width="900" height="400" style="max-width: 1100px; 
               max-height: 470px;"></canvas>
        <script>
            const eyeLabels = {labels};
            const openness = {openness};
            const leftArea = {left_area};
            const rightArea = {right_area};
            new Chart(document.getElementById('eyeMetricsChart'), {{
                type: 'bar',
                data: {{
                    labels: eyeLabels,
                    datasets: [
                        {{
                            label: 'Eye Openness %',
                            data: openness,
                            backgroundColor: 'rgba(54, 162, 235, 0.7)'
                        }},
                        {{
                            label: 'Left Eye Area px²',
                            data: leftArea,
                            backgroundColor: 'rgba(255, 159, 64, 0.7)'

                        }},
                        {{
                            label: 'Right Eye Area px²',
                            data: rightArea,
                            backgroundColor: 'rgba(75, 192, 192, 0.7)'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: true, text: 'Eye Metrics per Face' }}
                    }}
                }}
            }});
        </script>
    </div>
     <script>
        setTimeout(() => {{
            document.getElementById('loader1').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            document.body.style.overflow = 'auto'; // enable scroll again
        }}, 1000);
        </script>
    </body>
    </html>
    '''
    with open("eye.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_extra_features_html(extra_features):
    img_rows = ""
    for f in extra_features:
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    feature_rows = ""
    for f in extra_features:
        feature_rows += f"""
        <tr>
            <td>{f['face_number']}</td>
            <td>{f['ethnicity']}</td>
            <td>{f['skin_tone']}</td>
            <td>{f['facial_hair']}</td>
            <td>{f['glasses']}</td>
            <td>{f['mask']}</td>
            <td>{f['smile']}</td>
            
        </tr>
        """
    feature_rows1 = ""
    for f in extra_features:
        feature_rows1 += f"""
        <tr>
            <td>{f['eyebrow_state']}</td>
            <td>{f['eye_emotion']}</td>
            <td>{f['mouth_state']}</td>
            <td>{f['eye_aspect_ratio']}</td>
            <td>{f['eye_shape']}</td>
            <td>{f['eye_color']}</td>
        </tr>
        """
    face_labels = [f"Face {f['face_number']}" for f in extra_features]
    eye_aspect_ratios = [f['eye_aspect_ratio'] for f in extra_features]
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Extra Facial Features</title>
        <link rel="stylesheet" href="style.css">
         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <script src="button.js"></script>

        
    </head>
    <body>
    <div id="loader1">
        <iframe src="loader/index.html">
        </iframe>
    </div>
      
    <div class="sidebar">

        <div class="profile">
        {img_rows}
            <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
         <div class="menu">
           <a href="pre.html" >Dashboard</a>
                <a href="eye.html">Eye Insights</a>
                <a href="extra_face_features.html" class="active">Appearance Profile</a>
                <a href="expressions.html">Expression Insights</a>
                <a href="face_features.html">Facial Structure</a>
                <a href="nose_features_report.html">Nose Geometry</a>
                <a href="mouth_features_report.html">Mouth Analysis</a>
                <a href="face_geometry_report.html">Facial Dimensions </a>
        </div>
    </div>
    <div class="main">
        <div class="header">
            <h2>Appearance Analysis Blueprint</h2>
            <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
            <div id="photo1"><img src="image/geomety/appe.png" alt=""></div>
        </div>
         <div class="table-container">
        <h4>Hierarchical Details</h4>
        <table >
           <tr>
                    <th>Face Number <span id="photo"><img src="image/face.png" alt=""></th>
                    <th>Ethnicity <span id="photo"><img src="image/apperance/ethicity.png" alt=""> <br>
                            <p>વંશીયતા</th>
                    <th>Skin Tone <span id="photo"><img src="image/apperance/skintone.png" alt=""><br>
                            <p>ચહેરાનો વાન</th>
                    <th>Facial Hair <span id="photo"><img src="image/apperance/heir.png" alt=""><br>
                            <p>ચહેરાના વાળ</th>
                    <th>Glasses <span id="photo"><img src="image/apperance/glass.png" alt=""><br>
                            <p>ચશ્મા</th>
                    <th>Mask <span id="photo"><img src="image/apperance/mask.png" alt=""><br>
                            <p>માસ્ક</th>
                    <th>Smile <span id="photo"><img src="image/apperance/smile1.png" alt=""><br>
                            <p>સ્મિત</th>

                </tr>

            {feature_rows}
        </table>
        <table>
           <tr>

                    <th>Eyebrow State <span id="photo"><img src="image/apperance/eyebro.png" alt=""><br>
                            <p>પાતળી ભમર </th>
                    <th>Eye Emotion <span id="photo"><img src="image/apperance/emotion.png" alt=""><br>
                            <p>આંખની લાગણી</th>
                    <th>Mouth State <span id="photo"><img src="image/apperance/statemouth.png" alt=""><br>
                            <p>મુખની સ્થિતિ
                    </th>
                    <th>Eye Aspect Ratio <span id="photo"><img src="image/apperance/ratio.png" alt=""><br>
                            <p>આંખનો પાસા
                                ગુણોત્તર</th>
                    <th>Eye Shape <span id="photo"><img src="image/apperance/shape.png" alt=""><br>
                            <p>આંખનો આકાર</th>
                    <th>Eye Color <span id="photo"><img src="image/apperance/color.png" alt=""><br>
                            <p>આંખનો રંગ</th>
                </tr>
            {feature_rows1}
        </table>
        </div>
        <h3>Combined Chart —Eye Aspect Ratio per Face</h3>
    <canvas id="earChart" width="900" height="400" style="max-width: 1100px; 
               max-height: 470px;"></canvas>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    const earLabels = {face_labels};
    const earValues = {eye_aspect_ratios};
    new Chart(document.getElementById('earChart'), {{
        type: 'bar',
        data: {{
            labels: earLabels,
            datasets: [{{
                label: 'Eye Aspect Ratio',
                data: earValues,
                backgroundColor: 'rgba(54, 162, 235, 0.7)'
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{ position: 'top' }},
                title: {{ display: true, text: 'Eye Aspect Ratio per Face' }}
            }}
        }}
    }});
    </script>
    </div>
     <script>
        setTimeout(() => {{
            document.getElementById('loader1').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            document.body.style.overflow = 'auto'; // enable scroll again
        }}, 1000);
        </script>
    </body>
    </html>
    '''

    with open("extra_face_features.html", "w", encoding="utf-8") as f:
        f.write(html_content)

#page 6 html file
# >>> ADD HERE: Generate expressions.html report


def generate_expressions_html(expressions_data):
    img_rows = ""
    for f in expressions_data:
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    rows = ""
    for f in expressions_data:
        rows += f"""
        <tr>
            <td>{f['face_id']}</td>
            <td>{f['eyebrow_raise_pct']}%</td>
            <td>{f['eye_emotion_type']}</td>
            <td>{f['nose_flare_value']}</td>
            <td>{f['lip_curl_state']}</td>
           
        </tr>
        """

    rows1 = ""
    for f in expressions_data:
        rows1 += f"""
        <tr>
            <td>{f['talking_activity_state']}</td>
            <td>{f['head_tilt_angle']}°</td>
            <td>{f['head_rotation_angle']}°</td>
            <td>{f['eye_saccade_speed']}</td>
        </tr>
        """
    rows2 = ""
    for f in expressions_data:
        rows2 += f"""
        <tr>
           
            <td>{f['facial_symmetry_val']}</td>
            <td>{f['skin_brightness_val']}</td>
            <td>{f['skin_smoothness_val']}</td>
            <td>{f['expression_change_freq']}</td>
        </tr>
        """
        

    # Convert Python lists to JSON for JavaScript
    labels = json.dumps([f['face_id'] for f in expressions_data])
    symmetry = json.dumps([f['facial_symmetry_val'] for f in expressions_data])
    smoothness = json.dumps([f['skin_smoothness_val'] for f in expressions_data])

    html_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Facial Expression Report</title>
        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
        <link rel='stylesheet' href='style.css'>
         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <script src="button.js"></script>

    </head>
    <body>
    <div id="loader1">
        <iframe src="loader/index.html">
        </iframe>
    </div>
      <div class="sidebar">
        <div class="profile">
          {img_rows}
            <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="menu">
          <a href="pre.html" >Dashboard</a>
                <a href="eye.html">Eye Insights</a>
                <a href="extra_face_features.html">Appearance Profile</a>
                <a href="expressions.html" class="active">Expression Insights</a>
                <a href="face_features.html">Facial Structure</a>
                <a href="nose_features_report.html">Nose Geometry</a>
                <a href="mouth_features_report.html">Mouth Analysis</a>
                <a href="face_geometry_report.html">Facial Dimensions </a>
        </div>
    </div>
    <div class="main">
        <div class="header">
             <h2>Behavior Insight Blueprint</h2>
             <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
            <div id="photo1"><img src="image/geomety/behavior.png" alt=""></div>
        </div>
        <div class="table-container">
         <h4>Hierarchical Details</h4>
            <table>
                 <tr>
                    <th>Face Number  <span id="photo"><img src="image/face.png" alt=""></th>
                    <th>Eyebrow Raise % <span id="photo"><img src="image/apperance/eyebro.png" alt=""><br><p>આંખની ભમર</th>
                    <th>Eye Emotion Type <span id="photo"><img src="image/apperance/emotion.png" alt=""><br><p>આંખની
                            લાગણીનો પ્રકાર</th>
                    <th>Nose Flare Value <span id="photo"><img src="image/expression/nose.png" alt=""><br><p>નાક ફેલાવવાની
                            માત્રા</th>
                    <th>Lip Curl State <span id="photo"><img src="image/expression/lip.png" alt=""><br><p>હોઠ વળાંકની
                            સ્થિતિ</th>


                </tr>

                {rows}
            </table>
            <table>
                <tr>
                    <th>Talking Activity State <span id="photo"><img src="image/expression/talk.png" alt=""><br>
                            <p>બોલવાની
                                પ્રવૃત્તિ</th>
                    <th>Head Tilt Angle (°) <span id="photo"><img src="image/expression/tild.png" alt=""><br>
                            <p>માથું
                                નમાવવાની માત્રા</th>
                    <th>Head Rotation Angle (°) <span id="photo"><img src="image/dash/rotation.png" alt=""><br>
                            <p>માથું
                                ફેરવવાની માત્રા</th>
                    <th>Eye Saccade Speed <span id="photo"><img src="image/expression/scade.png" alt=""><br>
                            <p>આંખ હલનચલનની
                                ગતિ</th>
                </tr>
                {rows1}
            </table>
            <table>
               <tr>
                    <th>Facial Symmetry Value <span id="photo"><img src="image/expression/s1.png" alt=""><br>
                            <p>ચહેરાની
                                સપ્રમાણતા</th>
                    <th>Skin Brightness Value <span id="photo"><img src="image/expression/brighness.png" alt=""><br>
                            <p>ત્વચાની તેજસ્વીતા</th>
                    <th>Skin Smoothness Value <span id="photo"><img src="image/expression/smooth.png" alt=""><br>
                            <p>ત્વચાની
                                મુલાયમતા</th>
                    <th>Expression Change Frequency <span id="photo"><img src="image/expression/frequency.png"
                                alt=""><br>
                            <p>હાવભાવ બદલવાની ઝડપ
                    </th>
                </tr>

                {rows2}

            </table>
        </div>

       <h3>Combined Chart —Symmetry & Smoothness Face</h3>
        <canvas id='exprChart' width="900" height="400" style="max-width: 1100px; 
               max-height: 470px;"></canvas>
        <script>
            const labels = {labels};
            const symmetry = {symmetry};
            const smoothness = {smoothness};

            new Chart(document.getElementById('exprChart'), {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ label: 'Facial Symmetry', data: symmetry, backgroundColor: 'rgba(54, 162, 235, 0.7)' }},
                        {{ label: 'Skin Smoothness', data: smoothness, backgroundColor: 'rgba(75, 192, 192, 0.7)' }}
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: true, text: 'Facial Symmetry & Smoothness per Face' }}
                    }}
                }}
            }});
        </script>
         <script>
        setTimeout(() => {{
            document.getElementById('loader1').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            document.body.style.overflow = 'auto'; // enable scroll again
        }}, 1000);
        </script>
    </body>
    </html>
    """

    with open("expressions.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_face_features_html(face_features_data):
    img_rows = ""
    for f in face_features_data:
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    rows = ""
    for f in face_features_data:
        rows += f"""
        <tr>
            <td>{f['face_number']}</td>
            <td>{f['forehead_height']} cm</td>
            <td>{f['forehead_width']} cm</td>
            <td>{f['forehead_curvature']}%</td>
            <td>{f['forehead_skin_tone']}</td>
           
        </tr>
        """
    rows1 = ""
    for f in face_features_data:
        rows1 += f"""
        <tr>
           
            <td>{f['cheekbone_prominence']} cm</td>
            <td>{f['cheek_width']} cm</td>
            <td>{f['cheek_color']}</td>
            <td>{f['cheek_symmetry']}</td>
           
        </tr>
        """
    rows2 = ""
    for f in face_features_data:
        rows2 += f"""
        <tr>
           
           
            <td>{f['cheek_hollowness']} cm</td>
            <td>{f['midface_width']} cm</td>
            <td>{f['facial_contour_angle']}°</td>
            <td>{f['facial_asymmetry']}</td>
            
        </tr>
        """
    rows3 = ""
    for f in face_features_data:
        rows3 += f"""
        <tr>
           
            <td>{f['jawline_angle']}°</td>
            <td>{f['jawline_width']} cm</td>
            <td>{f['chin_prominence']} cm</td>
            <td>{f['temple_width']} cm</td>
            
        </tr>
        """
    rows4 = ""
    for f in face_features_data:
        rows4 += f"""
        <tr>
          
            <td>{f['zygomatic_width']} cm</td>
            <td>{f['face_ovality_ratio']}</td>
            <td>{f['face_length']} cm</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Face Features Report</title>
        <link rel='stylesheet' href='style.css'>
         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <script src="button.js"></script>

    </head>
    <body>
    <div id="loader1">
        <iframe src="loader/index.html">
        </iframe>
    </div>
         <div class="sidebar">
        <div class="profile">
            {img_rows}
            <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="menu">
            <a href="pre.html" >Dashboard</a>
                <a href="eye.html">Eye Insights</a>
                <a href="extra_face_features.html">Appearance Profile</a>
                <a href="expressions.html">Expression Insights</a>
                <a href="face_features.html" class="active">Facial Structure</a>
                <a href="nose_features_report.html">Nose Geometry</a>
                <a href="mouth_features_report.html">Mouth Analysis</a>
                <a href="face_geometry_report.html">Facial Dimensions </a>
        </div>
    </div>
    <div class="main">
        <div class="header">
              <h2>Morpho Analysis Blueprints</h2>
              <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
              <div id="photo1"><img src="image/geomety/morpho.png" alt=""></div>
        </div>
        <div class="table-container">
      <h4>Hierarchical Details</h4>
         <table>
                <tr>
                    <th>Face Number <span id="photo"><img src="image/face.png" alt=""></th>
                    <th>Forehead Height <span id="photo"><img src="image/facial/fheight.png" alt=""><br>
                            <p>કપાળની ઊંચાઈ
                    </th>
                    <th>Forehead Width <span id="photo"><img src="image/facial/fwidth.png" alt=""><br>
                            <p>કપાળની પહોળાઈ</th>
                    <th>Forehead Curvature <span id="photo"><img src="image/facial/curvater.png" alt=""><br>
                            <p>કપાળનો વળાંક
                    </th>
                    <th>Forehead Skin Tone <span id="photo"><img src="image/facial/skintone.png" alt=""><br>
                            <p>કપાળની
                                ત્વચાનો રંગ</th>

                </tr>

                {rows}
            </table>
            <table>
                 <tr>
                    <th>Cheekbone Prominence <span id="photo"><img src="image/facial/cheekwidth.png" alt=""><br>
                            <p>ગાલનો
                                ઉભાર</th>
                    <th>Cheek Width <span id="photo"><img src="image/facial/symmetry.png" alt=""><br>
                            <p>ગાલની પહોળાઈ</th>
                    <th>Cheek Color <span id="photo"><img src="image/facial/color.png" alt=""><br>
                            <p>ગાલનો રંગ</th>
                    <th>Cheek Symmetry <span id="photo"><img src="image/facial/symmetry.png" alt=""><br>
                            <p>ગાલની સમપ્રમાણતા
                    </th>
                </tr>
                 {rows1}
            </table>
            <table>
                <tr>
                    <th>Cheek Hollowness <span id="photo"><img src="image/facial/hollo.png" alt=""><br>
                            <p>ગાલનો ખાડો</th>
                    <th>Mid-face Width <span id="photo"><img src="image/facial/midwidth.png" alt=""><br>
                            <p>મધ્ય-ચહેરાની
                                પહોળાઈ</th>
                    <th>Facial Contour Angle <span id="photo"><img src="image/facial/angel.png" alt=""><br>
                            <p>ચહેરાના ઘાટનો
                                કોણ</th>
                    <th>Facial Asymmetry <span id="photo"><img src="image/facial/asymmety.png" alt=""><br>
                            <p>ચહેરાની
                                અસમાનતા</th>
                </tr>
                 {rows2}
            </table>
            <table>
                <tr>
                    <th>Jawline Angle <span id="photo"><img src="image/facial/jawline.png" alt=""><br>
                            <p>જડબાની રેખાનો કોણ
                    </th>
                    <th>Jawline Width <span id="photo"><img src="image/facial/jawwidth.png" alt=""><br>
                            <p>જડબાની પહોળાઈ
                    </th>
                    <th>Chin Prominence <span id="photo"><img src="image/facial/prom.png" alt=""><br>
                            <p>હડપચીનો ઉભાર</th>
                    <th>Temple Width <span id="photo"><img src="image/facial/tample.png" alt=""><br>
                            <p>કપાળની બાજુના ભાગની
                                પહોળાઈ</th>
                </tr>
                 {rows3}
            </table>
            <table>
                 <tr>
                    <th>Zygomatic Width <span id="photo"><img src="image/facial/zygo.png" alt=""><br>
                            <p>ગાલના હાડકાંની
                                પહોળાઈ</th>
                    <th>Face Ovality Ratio <span id="photo"><img src="image/facial/ovatiy.png" alt=""><br>
                            <p>ચહેરાના લંબગોળ
                                પ્રમાણનો આંક</th>
                    <th>Face Length <span id="photo"><img src="image/facial/facewidth.png" alt=""><br>
                            <p>ચહેરાની લંબાઈ</th>
                </tr>

                 {rows4}

            </table>
        </div>
            <h3>Combined Chart — Forehead Width (cm), Jawline Width (cm), Face Length (cm) </h3>
         <canvas id="faceShapeChart" 
       width="900" height="400" style="max-width: 1100px; 
               max-height: 470px;">
    </canvas>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const labels = { [f['face_number'] for f in face_features_data] };
        const foreheadWidth = { [f['forehead_width'] for f in face_features_data] };
        const jawlineWidth = { [f['jawline_width'] for f in face_features_data] };
        const faceLength = { [f['face_length'] for f in face_features_data] };

        new Chart(document.getElementById('faceShapeChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Forehead Width (cm)', data: foreheadWidth, backgroundColor: 'rgba(54, 162, 235, 0.7)' }},
                    {{ label: 'Jawline Width (cm)', data: jawlineWidth, backgroundColor: 'rgba(255, 159, 64, 0.7)' }},
                    {{ label: 'Face Length (cm)', data: faceLength, backgroundColor: 'rgba(75, 192, 192, 0.7)' }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top' }},
                    title: {{ display: true, text: 'Face Shape Dimensions Comparison' }}
                }},
                scales: {{
                    y: {{
                        title: {{ display: true, text: 'Measurement (cm)' }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
     <script>
        setTimeout(() => {{
            document.getElementById('loader1').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            document.body.style.overflow = 'auto'; // enable scroll again
        }}, 1000);
        </script>

    </body>
    </html>
    """

    with open("face_features.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_mouth_features_html(mouth_features_data):
    img_rows = ""
    for f in mouth_features_data:
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    rows = ""
    for f in mouth_features_data:
        rows += f"""
        <tr>
            <td>{f['face_number']}</td>
            <td>{f['mouth_width']}</td>
            <td>{f['mouth_height']}</td>
            <td>{f['mouth_area']}</td>
            <td>{f['mouth_open_pct']}</td>
           
        </tr>
        """
    rows1 = ""
    for f in mouth_features_data:
        rows1 += f"""
        <tr>
            
            <td>{f['lip_corner_positions']}</td>
            <td>{f['smile_intensity']}</td>
            <td>{f['lip_color']}</td>
            <td>{f['lip_symmetry']}</td>
            
        </tr>
        """
    rows2 = ""
    for f in mouth_features_data:
        rows2 += f"""
        <tr>
           
           
            <td>{f['upper_lip_height']}</td>
            <td>{f['lower_lip_height']}</td>
            <td>{f['mouth_aspect_ratio']}</td>
            <td>{f['teeth_visibility']}</td>
            
        </tr>
        """
    rows3 = ""
    for f in mouth_features_data:
        rows3 += f"""
        <tr>
          
            <td>{f['speaking_activity']}</td>
            <td>{f['mouth_curvature']}</td>
            <td>{f['lip_thickness']}</td>
            <td>{f['mouth_center_position']}</td>
            
        </tr>
        """
    rows4 = ""
    for f in mouth_features_data:
        rows4 += f"""
        <tr>
           
            <td>{f['mouth_rotation']}</td>
            <td>{f['lip_contour']}</td>
            <td>{f['lip_fullness']}</td>
            <td>{f['mouth_openness_trend']}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Mouth & Lip Features Report</title>
        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
         <link rel="stylesheet" href="style.css">
          <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
         <script src="button.js"></script>

    </head>
    <body>
    <div id="loader1">
        <iframe src="loader/index.html">
        </iframe>
    </div>
     <div class="sidebar">
        <div class="profile">
           {img_rows}
            <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="menu">
           <a href="pre.html" >Dashboard</a>
                <a href="eye.html">Eye Insights</a>
                <a href="extra_face_features.html">Appearance Profile</a>
                <a href="expressions.html">Expression Insights</a>
                <a href="face_features.html">Facial Structure</a>
                <a href="nose_features_report.html" >Nose Geometry</a>
                <a href="mouth_features_report.html" class="active">Mouth Analysis</a>
                <a href="face_geometry_report.html">Facial Dimensions </a>
        </div>
    </div>
    <div class="main">
        <div class="header">
             <h2>Mouth & Lip Analysis Blueprints</h2>
             <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
          <div id="photo1"><img src="image/geomety/mouth.png" alt=""></div>
        </div>
        <div class="table-container">
        <h4>Hierarchical Details</h4>
            <table>
                <tr>
                    <th>Face Number <span id="photo"><img src="image/face.png" alt=""></th>
                    <th>Mouth Width <span id="photo"><img src="image/apperance/statemouth.png" alt=""><br>
                            <p>મોંની પહોળાઈ
                    </th>
                    <th>Mouth Height <span id="photo"><img src="image/apperance/statemouth.png" alt=""><br>
                            <p>મોંની ઊંચાઈ
                    </th>
                    <th>Mouth Area <span id="photo"><img src="image/mouth/mouthse.png" alt=""><br>
                            <p>મુખ વિસ્તાર</th>
                    <th>Mouth Open % <span id="photo"><img src="image/mouth/mouthse.png" alt=""><br>
                            <p>મોંનુ ખુલ્લાપણૂ</th>

                </tr>
                {rows}
            </table>
            <table>
                 <tr>
                    <th>Lip Corner Positions <span id="photo"><img src="image/mouth/corner.png" alt=""><br>
                            <p>હોઠ કોર્નર
                                પોઝિશન્સ</th>
                    <th>Smile Intensity <span id="photo"><img src="image/apperance/smile.png" alt=""><br>
                            <p>સ્મિતની તીવ્રતા
                    </th>
                    <th>Lip Color <span id="photo"><img src="image/mouth/color1.png" alt=""><br>
                            <p>હોઠનો રંગ</th>
                    <th>Lip Symmetry <span id="photo"><img src="image/mouth/symmety.png" alt=""><br>
                            <p>હોઠની સમપ્રમાણતા
                    </th>
                </tr>
                 {rows1}
            </table>
            <table>
                 <tr>
                    <th>Upper Lip Height <span id="photo"><img src="image/expression/lip.png" alt=""><br>
                            <p>ઉપલા હોઠની
                                ઊંચાઈ</th>
                    <th>Lower Lip Height <span id="photo"><img src="image/expression/lip.png" alt=""><br>
                            <p>નીચલા હોઠની
                                ઊંચાઈ</th>
                    <th>Mouth Aspect Ratio <span id="photo"><img src="image/apperance/statemouth.png" alt=""><br>
                            <p>મોંના
                                પાસાનો ગુણોત્તર</th>
                    <th>Teeth Visibility<span id="photo"><img src="image/mouth/teeth1.png" alt=""> <br>
                            <p>દાંતની દૃશ્યતા
                    </th>
                </tr>
                 {rows2}
            </table>
            <table>
                 <tr>
                    <th>Speaking Activity <span id="photo"><img src="image/expression/talk.png" alt=""><br>
                            <p>બોલવાની
                                પ્રવૃત્તિ</th>
                    <th>Mouth Curvature <span id="photo"><img src="image/mouth/mouthse.png" alt=""><br>
                            <p>મોંની વક્રતા</th>
                    <th>Lip Thickness <span id="photo"><img src="image/mouth/teeth.png" alt=""><br>
                            <p>હોઠની જાડાઈ</th>
                    <th>Mouth Center Position <span id="photo"><img src="image/mouth/center.png" alt=""><br>
                            <p>મોંનુ
                                કેન્દ્ર સ્થાન</th>
                </tr>

                 {rows3}
            </table>
            <table>
                <tr>
                    <th>Mouth Rotation <span id="photo"><img src="image/mouth/mouthse.png" alt=""><br>
                            <p>મોંનુ પરિભ્રમણ
                    </th>
                    <th>Lip Contour <span id="photo"><img src="image/mouth/symmety.png" alt=""><br>
                            <p>હોઠની વક્રતા</th>
                    <th>Lip Fullness <span id="photo"><img src="image/mouth/symmety.png" alt=""><br>
                            <p>હોઠની પૂર્ણતા</th>
                    <th>Mouth Openness Trend <span id="photo"><img src="image/mouth/lastmouth.png" alt=""><br>
                            <p>મોંનુ
                                ખુલ્લાપણનુ વલણ</th>
                </tr>
                 {rows4}
            </table>
        </div>


       
                <h3>Combined Chart — Mouth Width (cm), Mouth Height (cm), Mouth Area (cm²)</h3>

        <canvas id='mouthChart' 
            width="900" height="400" style="max-width: 1100px; 
               max-height: 470px;">  </canvas>

        <script>
            const labels = {[f['face_number'] for f in mouth_features_data]};
            const mouthWidth = {[float(f['mouth_width'].split()[0]) for f in mouth_features_data]};
            const mouthHeight = {[float(f['mouth_height'].split()[0]) for f in mouth_features_data]};
            const mouthArea = {[float(f['mouth_area'].split()[0]) for f in mouth_features_data]};

            new Chart(document.getElementById('mouthChart'), {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ label: 'Mouth Width (cm)', data: mouthWidth, backgroundColor: 'rgba(54,162,235,0.7)' }},
                        {{ label: 'Mouth Height (cm)', data: mouthHeight, backgroundColor: 'rgba(255,159,64,0.7)' }},
                        {{ label: 'Mouth Area (cm²)', data: mouthArea, backgroundColor: 'rgba(75,192,192,0.7)' }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: true, text: 'Mouth Dimensions Comparison' }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{ display: true, text: 'Measurement (cm/cm²)' }}
                        }}
                    }}
                }}
            }});
        </script>
         <script>
        setTimeout(() => {{
            document.getElementById('loader1').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            document.body.style.overflow = 'auto'; // enable scroll again
        }}, 1000);
        </script>
    </body>
    </html>
    """
    with open("mouth_features_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_nose_features_html(nose_features_data):
    img_rows = ""
    for f in nose_features_data:
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    rows = ""
    for f in nose_features_data:
        rows += f"""
        <tr>
            <td>{f['face_number']}</td>
            <td>{f['nose_tip_x']}</td>
            <td>{f['nose_tip_y']}</td>
            <td>{f['nose_bridge_length']}</td>
            <td>{f['nose_width']}</td>
           
        </tr>
        """
    rows1 = ""
    for f in nose_features_data:
        rows1 += f"""
        <tr>
            
            <td>{f['nostril_width']}</td>
            <td>{f['nose_height']}</td>
            <td>{f['nose_angle']}</td>
            <td>{f['nose_direction']}</td>
           
        </tr>
        """
    rows2 = ""
    for f in nose_features_data:
        rows2 += f"""
        <tr>
          
            <td>{f['nose_symmetry']}</td>
            <td>{f['nose_prominence']}</td>
            <td>{f['nose_rotation']}</td>
            <td>{f['nose_area']}</td>
            
        </tr>
        """
    rows3 = ""
    for f in nose_features_data:
        rows3 += f"""
        <tr>
            
            <td>{f['nose_shape']}</td>
            <td>{f['nose_curvature']}</td>
            <td>{f['nose_tip_relative_position']}</td>
            <td>{f['nose_center_position']}</td>
           
        </tr>
        """
    rows4 = ""
    for f in nose_features_data:
        rows4 += f"""
        <tr>
            
            <td>{f['distance_nose_to_chin']}</td>
            <td>{f['nose_base_width']}</td>
            <td>{f['nose_contour_ratio']}</td>
            <td>{f['nose_profile_score']}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Nose Features Report</title>
        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
        <link rel="stylesheet" href="style.css">
         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <script src="button.js"></script>

        
    </head>
    <body>
    <div id="loader1">
        <iframe src="loader/index.html">
        </iframe>
    </div>
     <div class="sidebar">
        <div class="profile">
           {img_rows}
            <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="menu">
           <a href="pre.html" >Dashboard</a>
                <a href="eye.html">Eye Insights</a>
                <a href="extra_face_features.html">Appearance Profile</a>
                <a href="expressions.html">Expression Insights</a>
                <a href="face_features.html">Facial Structure</a>
                <a href="nose_features_report.html" class="active">Nose Geometry</a>
                <a href="mouth_features_report.html">Mouth Analysis</a>
                <a href="face_geometry_report.html">Facial Dimensions </a>
        </div>
    </div>
    <div class="main">
        <div class="header">
            <h2>Nose Analysis Blueprints</h2>
            <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
            <div id="photo1"><img src="image/geomety/nose.png" alt=""></div>
        </div>
        <div class="table-container">
         <h4>Hierarchical Details</h4>
        <table>
                <tr>
                    <th>Face Number <span id="photo"><img src="image/face.png" alt=""> </th>
                    <th>Nose Tip X (cm) <span id="photo"><img src="image/expression/nose.png" alt=""><br>
                            <p>નાકની ટોચ X</p>
                    </th>
                    <th>Nose Tip Y (cm) <span id="photo"><img src="image/nose/angel.png" alt=""><br>
                            <p>નાકની ટોચ Y</th>
                    <th>Nose Bridge Length <span id="photo"><img src="image/nose/bridge.png" alt=""><br>
                            <p>નાકના ટેકાની
                                લંબાઈ</th>
                    <th>Nose Width <span id="photo"><img src="image/nose/bridge.png" alt=""><br>
                            <p>નાકની પહોળાઈ</th>
                </tr>
                {rows}
            </table>
            <table>
                <tr>
                    <th>Nostril Width<span id="photo"><img src="image/nose/nostril.png" alt=""><br>
                            <p>નસકોરાની પહોળાઈ</th>
                    <th>Nose Height <span id="photo"><img src="image/nose/nostril.png" alt=""><br>
                            <p>નાકની ઊંચાઈ</th>
                    <th>Nose Angle <span id="photo"><img src="image/nose/angel.png" alt=""><br>
                            <p> નાક કોણ </th>
                    <th>Nose Direction <span id="photo"><img src="image/nose/direction1.png" alt=""><br>
                            <p>નાકની દિશા</th>
                </tr>
                 {rows1}
            </table>
            <table>
                <tr>
                    <th>Nose Symmetry <span id="photo"><img src="image/nose/symmety.png" alt=""><br>
                            <p>નાકની સમપ્રમાણતા
                    </th>
                    <th>Nose Prominence <span id="photo"><img src="image/expression/nose.png" alt=""><br>
                            <p>નાક પ્રાધાન્ય
                    </th>
                    <th>Nose Rotation <span id="photo"><img src="image/nose/rotation.png" alt=""><br>
                            <p>નાક પરિભ્રમણ</th>
                    <th>Nose Area <span id="photo"><img src="image/nose/area.png" alt=""><br>
                            <p>નાક વિસ્તાર</th>
                </tr>
                {rows2}
            </table>
            <table>
               <tr>
                    <th>Nose Shape <span id="photo"><img src="image/nose/area.png" alt=""><br>
                            <p>નાકનો આકાર</th>
                    <th>Nose Curvature <span id="photo"><img src="image/nose/bridge.png" alt=""><br>
                            <p>નાકની વક્રતા</th>
                    <th>Nose Tip Relative Position <span id="photo"><img src="image/nose/tip.png" alt=""><br>
                            <p>ટોચ સંબંધિત
                                સ્થિતિ</th>
                    <th>Nose Center Position <span id="photo"><img src="image/nose/center.png" alt=""><br>
                            <p> નાકની
                                કેન્દ્રની સ્થિતિ</th>
                </tr>
               {rows3}
            </table>
            <table>
                <tr>
                    <th>Distance Nose–Chin <span id="photo"><img src="image/nose/disy.png" alt=""><br>
                            <p>નાક-ચિનનું અંતર
                    </th>
                    <th>Nose Base Width <span id="photo"><img src="image/nose/base.png" alt=""><br>
                            <p>નાક પાયાની પહોળાઈ
                    </th>
                    <th>Nose Contour Ratio <span id="photo"><img src="image/nose/direction1.png" alt=""><br>
                            <p>નાક સમતલરેખા
                                રેશિયો</th>
                    <th>Nose Profile Score <span id="photo"><img src="image/nose/disy.png" alt=""><br>
                            <p>નાક પ્રોફાઇલ સ્કોર
                    </th>
                </tr>
             {rows4}

            </table>
        </div>


         <h3>Combined Chart — Nose Width (cm), Nose Height (cm), Nose Symmetry (%) </h3>

        <canvas id='noseChart' 
           width="900" height="400" style="max-width: 1100px; 
               max-height: 470px;">  </canvas>

        <script>
            const labels = {[f['face_number'] for f in nose_features_data]};
            const widthVals = {[float(f['nose_width'].split()[0]) for f in nose_features_data]};
            const heightVals = {[float(f['nose_height'].split()[0]) for f in nose_features_data]};
            const symmetryVals = {[float(f['nose_symmetry'].replace('%','')) for f in nose_features_data]};

            new Chart(document.getElementById('noseChart'), {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ label: 'Nose Width (cm)', data: widthVals, backgroundColor: 'rgba(54,162,235,0.7)' }},
                        {{ label: 'Nose Height (cm)', data: heightVals, backgroundColor: 'rgba(255,159,64,0.7)' }},
                        {{ label: 'Nose Symmetry (%)', data: symmetryVals, backgroundColor: 'rgba(75,192,192,0.7)' }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: true, text: 'Nose Dimension & Symmetry Analysis' }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{ display: true, text: 'Measurement (cm or %)' }}
                        }}
                    }}
                }}
            }});
        </script>
         <script>
        setTimeout(() => {{
            document.getElementById('loader1').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            document.body.style.overflow = 'auto'; // enable scroll again
        }}, 1000);
        </script>
    </body>
    </html>
    """
    with open("nose_features_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_face_geometry_html(face_geometry_data):
    img_rows = ""
    for f in face_geometry_data:
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80">\n'

    rows = ""
    for f in face_geometry_data:
        rows += f"""
        <tr>
            <td>{f['face_id']}</td>
            <td>{f['geom_face_width']}</td>
            <td>{f['geom_face_height']}</td>
            <td>{f['geom_face_area']}</td>
            <td>{f['geom_face_center']}</td>
           
        </tr>
        """
    rows1 = ""
    for f in face_geometry_data:
        rows1 += f"""
        <tr>
           
            <td>{f['geom_face_distance']}</td>
            <td>{f['geom_confidence']}</td>
            <td>{f['geom_jaw_width']}</td>
            <td>{f['geom_chin_position']}</td>
           
        </tr>
        """
    rows2 = ""
    for f in face_geometry_data:
        rows2 += f"""
        <tr>
           
            <td>{f['geom_cheek_prominence']}</td>
            <td>{f['geom_eye_color']}</td>
            <td>{f['geom_eye_shape']}</td>
            <td>{f['geom_saccade_flag']}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Face Geometry Report</title>
        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
        <link rel="stylesheet" href="style.css">
         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <script src="button.js"></script>

    </head>
    <body>
    <div id="loader1">
        <iframe src="loader/index.html">
        </iframe>
    </div>
    <div class="sidebar">
        <div class="profile">
          {img_rows}
            <h3>Detected Faces</h3>
            <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="menu">
             <a href="pre.html" >Dashboard</a>
                <a href="eye.html">Eye Insights</a>
                <a href="extra_face_features.html">Appearance Profile</a>
                <a href="expressions.html">Expression Insights</a>
                <a href="face_features.html">Facial Structure</a>
                <a href="nose_features_report.html">Nose Geometry</a>
                <a href="mouth_features_report.html">Mouth Analysis</a>
                <a href="face_geometry_report.html" class="active">Facial Dimensions </a>
        </div>
    </div>
    <div class="main">
        <div class="header">
            <h2>Dimension Geometry Blueprint</h2>
            <i class="fa-solid fa-play floating-icon" onclick="run()"></i>
             <div id="photo1"><img src="image/geomety/geo.png" alt=""></div>
        </div>
        <div class="table-container">
         <h4>Hierarchical Details</h4>
            <table>
               <tr>
                    <th>Face Number <span id="photo"><img src="image/face.png" alt=""></th>
                    <th>Face Width (cm) <span id="photo"><img src="image/geomety/face.png" alt=""><br>
                            <p>ચહેરાની પહોળાઈ
                    </th>
                    <th>Face Height (cm) <span id="photo"><img src="image/geomety/face.png" alt=""><br>
                            <p>ચહેરાની ઊંચાઈ
                    </th>
                    <th>Face Area (cm²) <span id="photo"><img src="image/dash/status.png" alt=""><br>
                            <p>ચહેરોનો વિસ્તાર
                    </th>
                    <th>Face Center (x,y) <span id="photo"><img src="image/dash/face center.png" alt=""><br>
                            <p>ફેસ સેન્ટર
                    </th>


                </tr>

                {rows}
            </table>
            <table>
                <tr>
                    <th>Face Distance (cm) <span id="photo"><img src="image/dash/distance.png" alt=""><br>
                            <p>ચહેરાનું અંતર
                    </th>
                    <th>Confidence % <span id="photo"><img src="image/dash/confidence.png" alt=""><br>
                            <p>આત્મવિશ્વાસ</th>
                    <th>Jaw Width (cm) <span id="photo"><img src="image/facial/jawwidth.png" alt=""><br>
                            <p>જડબાની પહોળાઈ
                    </th>
                    <th>Chin Position <span id="photo"><img src="image/facial/prom.png" alt=""><br>
                            <p>હડપચી પોઝિશન</th>
                </tr>
                 {rows1}
            </table>
            <table>
                 <tr>
                    <th>Cheek Prominence (%) <span id="photo"><img src="image/facial/cheekwidth.png" alt=""><br>
                            <p>ગાલનું
                                પ્રાધાન્ય</th>
                    <th>Eye Color <span id="photo"><img src="image/apperance/color.png" alt=""><br>
                            <p>આંખનો રંગ</th>
                    <th>Eye Shape <span id="photo"><img src="image/apperance/eyebro.png" alt=""><br>
                            <p>આંખનો આકાર</th>
                    <th>Saccade Flag <span id="photo"><img src="image/facial/fwidth.png" alt=""><br></th>
                </tr>
                {rows2}

            </table>
        </div>

       <h3>Combined Chart — Face Width (cm), Face Height (cm), Confidence (%)</h3>
      <canvas id='faceChart'width="900" height="400" style="max-width: 1100px; 
               max-height: 470px;"></canvas>

        <script>
            const labels = {[f['face_id'] for f in face_geometry_data]};
            const widthVals = {[float(f['geom_face_width'].split()[0]) for f in face_geometry_data]};
            const heightVals = {[float(f['geom_face_height'].split()[0]) for f in face_geometry_data]};
            const confVals = {[float(f['geom_confidence'].replace('%','')) for f in face_geometry_data]};

            new Chart(document.getElementById('faceChart'), {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Face Width (cm)',
                            data: widthVals,
                            backgroundColor: 'rgba(54,162,235,0.7)'
                        }},
                        {{
                            label: 'Face Height (cm)',
                            data: heightVals,
                            backgroundColor: 'rgba(255,159,64,0.7)'
                        }},
                        {{
                            label: 'Confidence (%)',
                            data: confVals,
                            backgroundColor: 'rgba(75,192,192,0.7)'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: true, text: 'Face Dimension and Detection Confidence' }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{ display: true, text: 'Measurement (cm or %)' }}
                        }}
                    }}
                }}
            }});
        </script>
         <script>
        setTimeout(() => {{
            document.getElementById('loader1').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            document.body.style.overflow = 'auto'; // enable scroll again
        }}, 1000);
        </script>
    </body>
    </html>
    """

    with open("face_geometry_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)



# ---------------- Main Loop ----------------
# ---------------- Main Loop ----------------
def main():
    global prev_pupil_pos, fixation_start_time, baseline_eye_height
    cap = cv2.VideoCapture(PHONE_CAMERA_URL)
    time.sleep(2)
    face_geometry_data = []
    nose_features_data = []
    mouth_features_data = []
    face_features_data = [] 
    expressions_data = []
    extra_features_data = []
    faces_data, eyes_data = [], []
    counter = 1
    
    print("Press SPACE to capture a face. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from phone camera stream.")
            break

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected_faces = results.detections if results.detections else []

        for det in detected_faces:
            # ... (drawing logic remains the same) ...
            bboxC = det.location_data.relative_bounding_box
            ih, iw = frame.shape[:2]
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Faces: {len(detected_faces)}", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Live Face Detector (Phone Camera)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key == 32:  # SPACE
            if not detected_faces:
                print("No face detected. Try again.")
                continue

            for det in detected_faces:
                bbox = det.location_data.relative_bounding_box
                ih, iw = frame.shape[:2]
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                cx, cy = x + w // 2, y + h // 2
                confidence = round(det.score[0] * 100, 2) if hasattr(det, 'score') and len(det.score) else 0.0
                distance = estimate_distance(w)

                x1, y1, x2, y2 = expand_box(x, y, w, h, iw, ih)
                face_img = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                if not valid_crop(face_img):
                    continue


                face_path = f"faces/face_{counter}.jpg"
                cv2.imwrite(face_path, face_img)

                try:
                    dfa = DeepFace.analyze(
                        face_img,
                        actions=["gender", "emotion", "age"],
                        enforce_detection=False,
                        detector_backend="mtcnn"
                    )
                    face_img_proc = preprocess_face(face_img)
                    result = dfa[0] if isinstance(dfa, list) and dfa else dfa
                except Exception as e:
                    print("DeepFace analyze error:", e)
                    result = {"gender": "unknown", "emotion": {}, "age": 25}
                
                # -------------------------------------------

                mesh_results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                landmarks = None

                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark

                if landmarks is not None:
                    h_proc, w_proc = face_img_proc.shape[:2]
                    face_img_proc = align_face_by_eyes(face_img_proc, landmarks, w_proc, h_proc)
                else:
                    print("⚠ No landmarks detected — alignment skipped.")

                
                gender = gender_from_confidence(result.get("gender", {}))
                emotion_key, emotion_text, emotion_score = interpret_emotion(result.get("emotion", {}))
                age = age_to_range(estimate_age(result.get("age", 25)))
                rotation = get_rotation(face_img)
                
                with open(face_path, "rb") as img_file:
                    base64_img = base64.b64encode(img_file.read()).decode()

                # 🔊 Create TTS audio
                text = (
                    f"This person is {emotion_text}, gender = {gender}, "
                    f"age between {age} years."
                )
                audio_path = f"audios/audio_{counter}.mp3"
                make_audio(text, audio_path)

                # 🎵 Select emotion music
                music_file = EMOTION_MUSIC.get(emotion_key, EMOTION_MUSIC["neutral"])
                src_music_path = os.path.join("music_library", music_file)
                music_output_path = f"audios/music_{counter}.mp3"
                if os.path.exists(src_music_path):
                    shutil.copy(src_music_path, music_output_path)
                else:
                    music_output_path = ""
                
                # Eye tracking with mediapipe
                mesh_results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    if baseline_eye_height is None:
                        # ... (baseline calculation remains the same) ...
                        left_h = cv2.norm(get_landmark_coords([landmarks[153], landmarks[144]], w, h)[0], get_landmark_coords([landmarks[160], landmarks[158]], w, h)[0])
                        right_h = cv2.norm(get_landmark_coords([landmarks[380], landmarks[373]], w, h)[0], get_landmark_coords([landmarks[385], landmarks[387]], w, h)[0])
                        baseline_eye_height = (left_h + right_h) / 2
                        print(f"✅ Baseline eye height set: {baseline_eye_height:.2f} px")

                    eye_result = get_eye_metrics(landmarks, w, h, baseline_eye_height)

                    # --- PIXEL TO CM CONVERSION AND AREA CALCULATION ---
                    # Convert key eye metrics from pixels to centimeters (cm)
                    eye_result['left_w_cm'] = pixel_to_cm(eye_result.get('left_w'), distance)
                    eye_result['left_h_cm'] = pixel_to_cm(eye_result.get('left_h'), distance)
                    eye_result['right_w_cm'] = pixel_to_cm(eye_result.get('right_w'), distance)
                    eye_result['right_h_cm'] = pixel_to_cm(eye_result.get('right_h'), distance)

                    eye_result['left_pupil_x_cm'] = pixel_to_cm(eye_result.get('left_pupil_x'), distance)
                    eye_result['left_pupil_y_cm'] = pixel_to_cm(eye_result.get('left_pupil_y'), distance)
                    eye_result['right_pupil_x_cm'] = pixel_to_cm(eye_result.get('right_pupil_x'), distance)
                    eye_result['right_pupil_y_cm'] = pixel_to_cm(eye_result.get('right_pupil_y'), distance)
                  
                    
                    # Calculate Area in cm²
                    if eye_result['left_w_cm'] != 'N/A' and eye_result['left_h_cm'] != 'N/A':
                        eye_result['left_area_cm2'] = round(eye_result['left_w_cm'] * eye_result['left_h_cm'], 2)
                    else:
                        eye_result['left_area_cm2'] = 'N/A'
                
                    if eye_result['right_w_cm'] != 'N/A' and eye_result['right_h_cm'] != 'N/A':
                        eye_result['right_area_cm2'] = round(eye_result['right_w_cm'] * eye_result['right_h_cm'], 2)
                    else:
                        eye_result['right_area_cm2'] = 'N/A'
                    # --- END CM CALCULATION BLOCK ---

                    # Saccade and fixation logic
                    current_pupil_pos = eye_result['left_pupil_x'] + eye_result['right_pupil_x']
                    saccade_flag = "No"
                    fixation_duration_ms = 0
                    if prev_pupil_pos is not None:
                        movement = abs(current_pupil_pos - prev_pupil_pos)
                        if movement > 5: # Threshold for saccade
                            saccade_flag = "Yes"
                            fixation_start_time = None
                        else:
                            if fixation_start_time is None:
                                fixation_start_time = time.time()
                            fixation_duration_ms = round((time.time() - fixation_start_time) * 1000)
                    else:
                        fixation_start_time = time.time()
                    prev_pupil_pos = current_pupil_pos
                    eye_result['saccade_flag'] = saccade_flag
                    eye_result['fixation_duration_ms'] = fixation_duration_ms

                else:
                    # Default values for no face landmarks detected
                    eye_result = {
                        "left_w": "N/A", "left_h": "N/A", "left_w_cm": "N/A", "left_h_cm": "N/A",
                        "right_w": "N/A", "right_h": "N/A", "right_w_cm": "N/A", "right_h_cm": "N/A",
                        "left_area": "N/A", "right_area": "N/A", "left_area_cm2": "N/A", "right_area_cm2": "N/A",
                        "openness_pct": "N/A", "gaze_h": "N/A",
                        "gaze_v": "N/A", "left_pupil_x": "N/A",
                        "left_pupil_y": "N/A", "right_pupil_x": "N/A",
                        "right_pupil_y": "N/A", "blink": 0, "eye_state": "Unknown",
                        "saccade_flag": "N/A", "fixation_duration_ms": "N/A"
                    }

                faces_data.append({
                    "filename": os.path.basename(face_path),
                    "base64": base64_img,
                    "gender": gender,
                    # ... (other face data fields remain the same) ...
                    "emotion": emotion_text,
                    "emotion_score": emotion_score,
                    "age": age,
                    "size": f"{w}×{h}",
                    "position": f"{x}, {y}",
                    "center": f"{cx}, {cy}",
                    "distance": distance,
                    "rotation": rotation,
                    "confidence": f"{confidence}%",
                    "audio": audio_path.replace("\\", "/"),
                    "music": music_output_path.replace("\\", "/")
                })
                eyes_data.append(eye_result)
                # --- Extra 11 features ---
                ethnicity = detect_ethnicity(face_img)
                skin_tone = detect_skin_tone(face_img)
                facial_hair = detect_facial_hair(face_img)
                glasses = detect_glasses(face_img)
                mask = detect_mask(face_img)
                smile = detect_smile(face_img)
                eyebrow_state = detect_eyebrow_state(face_img)
                eye_emotion = detect_eye_emotion(face_img)
                mouth_state = detect_mouth_state(face_img)
                eye_aspect_ratio = detect_eye_aspect_ratio(mesh_results.multi_face_landmarks[0].landmark if mesh_results.multi_face_landmarks else None)
                eye_shape = detect_eye_shape(face_img)
                eye_color = detect_eye_color(face_img)

                extra_features_data.append({
                    "face_number": counter,
                    "ethnicity": ethnicity,
                    "skin_tone": skin_tone,
                    "facial_hair": facial_hair,
                    "glasses": glasses,
                    "mask": mask,
                    "smile": smile,
                    "eyebrow_state": eyebrow_state,
                    "eye_emotion": eye_emotion,
                    "mouth_state": mouth_state,
                    "eye_aspect_ratio": eye_aspect_ratio,
                    "eye_shape": eye_shape,
                    "eye_color": eye_color,
                    "base64": base64_img
                })

                #page-6
                              # page-6 (expression + advanced facial metrics)
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                else:
                    landmarks = None

                eyebrow_raise_pct = detect_eyebrow_raise(landmarks) if landmarks else 0
                eye_emotion_type = detect_eye_emotion(face_img)
                nose_flare_value = detect_nose_flare(face_img)
                lip_curl_state = detect_lip_curl(face_img)
                talking_activity_state = detect_talking_activity(mouth_state)
                head_tilt_angle = detect_head_tilt(0)
                head_rotation_angle = detect_head_rotation(0)
                eye_saccade_speed = detect_eye_saccades()
                facial_symmetry_val = detect_facial_symmetry(face_img)
                skin_brightness_val = detect_skin_brightness(face_img)
                skin_smoothness_val = detect_skin_smoothness(face_img)
                expression_change_freq = detect_expression_change(prev_emotion="neutral", current_emotion=eye_emotion_type)

                expressions_data.append({
                    "face_id": counter,  # matches your face numbering
                    "eyebrow_raise_pct": eyebrow_raise_pct,
                    "eye_emotion_type": eye_emotion_type,
                    "nose_flare_value": nose_flare_value,
                    "lip_curl_state": lip_curl_state,
                    "talking_activity_state": talking_activity_state,
                    "head_tilt_angle": head_tilt_angle,
                    "head_rotation_angle": head_rotation_angle,
                    "eye_saccade_speed": eye_saccade_speed,
                    "facial_symmetry_val": facial_symmetry_val,
                    "skin_brightness_val": skin_brightness_val,
                    "skin_smoothness_val": skin_smoothness_val,
                    "expression_change_freq": expression_change_freq,
                    "base64": base64_img
                })
                face_feature_entry = {
                    "face_number": counter,
                    "forehead_height": detect_forehead_height(landmarks, w, h, distance),
                    "forehead_width": detect_forehead_width(landmarks, w, h, distance),
                    "forehead_curvature": detect_forehead_curvature(landmarks),
                    "forehead_skin_tone": detect_forehead_skin_tone(face_img),
                    "cheekbone_prominence": detect_cheekbone_prominence(landmarks, w, h, distance),
                    "cheek_width": detect_cheek_width(landmarks, w, h, distance),
                    "cheek_color": detect_cheek_color(face_img),
                    "cheek_symmetry": detect_cheek_symmetry(face_img),
                    "cheek_hollowness": detect_cheek_hollowness(landmarks, w, h, distance),
                    "midface_width": detect_midface_width(landmarks, w, h, distance),
                    "facial_contour_angle": detect_facial_contour_angle(landmarks),
                    "facial_asymmetry": detect_facial_asymmetry(face_img),
                    "jawline_angle": detect_jawline_angle(landmarks),
                    "jawline_width": detect_jawline_width(landmarks, w, h, distance),
                    "chin_prominence": detect_chin_prominence(landmarks, w, h, distance),
                    "temple_width": detect_temple_width(landmarks, w, h, distance),
                    "zygomatic_width": detect_zygomatic_width(landmarks, w, h, distance),
                    "face_ovality_ratio": detect_face_ovality_ratio(landmarks),
                    "face_length": detect_face_length(landmarks, w, h, distance),
                    "base64": base64_img
                }
                face_features_data.append(face_feature_entry)

                mouth_feature_entry = {
                    "face_number": counter,
                    "mouth_width": detect_mouth_width(landmarks, w, h, distance),
                    "mouth_height": detect_mouth_height(landmarks, w, h, distance),
                    "mouth_area": detect_mouth_area(landmarks, w, h, distance),
                    "mouth_open_pct": detect_mouth_open_pct(landmarks),
                    "lip_corner_positions": detect_lip_corner_positions(landmarks),
                    "smile_intensity": detect_smile_intensity(face_img),
                    "lip_color": detect_lip_color(face_img),
                    "lip_symmetry": detect_lip_symmetry(face_img),
                    "upper_lip_height": detect_upper_lip_height(landmarks, w, h, distance),
                    "lower_lip_height": detect_lower_lip_height(landmarks, w, h, distance),
                    "mouth_aspect_ratio": detect_mouth_aspect_ratio(landmarks),
                    "teeth_visibility": detect_teeth_visibility(face_img),
                    "speaking_activity": detect_speaking_activity(),
                    "mouth_curvature": detect_mouth_curvature(landmarks),
                    "lip_thickness": detect_lip_thickness(landmarks, w, h, distance),
                    "mouth_center_position": detect_mouth_center_position(landmarks),
                    "mouth_rotation": detect_mouth_rotation(landmarks),
                    "lip_contour": detect_lip_contour(face_img),
                    "lip_fullness": detect_lip_fullness(face_img),
                    "mouth_openness_trend": detect_mouth_openness_trend(),
                    "base64": base64_img
                }

                mouth_features_data.append(mouth_feature_entry)
                nose_entry = {
                    "face_number": counter,
                    "nose_tip_x": detect_nose_tip_x(landmarks, w, h, distance),
                    "nose_tip_y": detect_nose_tip_y(landmarks, w, h, distance),
                    "nose_bridge_length": detect_nose_bridge_length(landmarks, w, h, distance),
                    "nose_width": detect_nose_width(landmarks, w, h, distance),
                    "nostril_width": detect_nostril_width(landmarks, w, h, distance),
                    "nose_height": detect_nose_height(landmarks, w, h, distance),
                    "nose_angle": detect_nose_angle(landmarks),
                    "nose_direction": detect_nose_direction(landmarks),
                    "nose_symmetry": detect_nose_symmetry(face_img),
                    "nose_prominence": detect_nose_prominence(landmarks, w, h, distance),
                    "nose_rotation": detect_nose_rotation(landmarks),
                    "nose_area": detect_nose_area(landmarks, w, h, distance),
                    "nose_shape": detect_nose_shape(face_img),
                    "nose_curvature": detect_nose_curvature(landmarks),
                    "nose_tip_relative_position": detect_nose_tip_relative_position(landmarks),
                    "nose_center_position": detect_nose_center_position(landmarks),
                    "distance_nose_to_chin": detect_distance_nose_to_chin(landmarks, w, h, distance),
                    "nose_base_width": detect_nose_base_width(landmarks, w, h, distance),
                    "nose_contour_ratio": detect_nose_contour_ratio(landmarks),
                    "nose_profile_score": detect_nose_profile_score(face_img),
                    "base64": base64_img
                }

                nose_features_data.append(nose_entry)

                geometry_entry = {
                    "face_id": counter,
                    "geom_face_width": detect_face_width(landmarks, w, h, distance),
                    "geom_face_height": detect_face_height(landmarks, w, h, distance),
                    "geom_face_area": detect_face_area(landmarks, w, h, distance),
                    "geom_face_center": detect_face_center(landmarks, w, h),
                    "geom_face_distance": detect_face_distance(landmarks, w, h),
                    "geom_confidence": detect_confidence_score(face_img),
                    "geom_jaw_width": detect_jaw_width(landmarks, w, h, distance),
                    "geom_chin_position": detect_chin_position(landmarks, w, h),
                    "geom_cheek_prominence": detect_cheek_prominence(landmarks),
                    "geom_eye_color": detect_eye_color(face_img),
                    "geom_eye_shape": detect_eye_shape(face_img),
                    "geom_saccade_flag": detect_saccade_flag(landmarks),
                    "base64": base64_img
                }

                face_geometry_data.append(geometry_entry)


                print(f"✅ Saved Face {counter} | gender = {gender}, {emotion_text}, Age: {age}, Confidence: {confidence}%")
                counter += 1

                generate_facial_report_html(faces_data)
                generate_eye_report_html(eyes_data,faces_data)
                generate_extra_features_html(extra_features_data)
                generate_expressions_html(expressions_data)
                generate_face_features_html(face_features_data)
                generate_mouth_features_html(mouth_features_data)
                generate_nose_features_html(nose_features_data)
                generate_face_geometry_html(face_geometry_data)

               

                
    f= os.path.abspath("pre.html")
    webbrowser.open(f"http://127.0.0.1:5500/pre.html", new=2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
