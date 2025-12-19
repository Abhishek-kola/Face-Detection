import cv2
import os
import base64
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp
import pyttsx3
import random
import shutil

# Phone camera streaming URL
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
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Eye landmark indices from MediaPipe's face mesh model.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

KNOWN_FACE_WIDTH_CM = 16
FOCAL_LENGTH_PIXELS = 800

#---------------- Eye Measurement ----------------
def get_eye_size(eye_points, landmarks, img_w, img_h):
    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_points]
    width = distance(points[0], points[3])
    height = (distance(points[1], points[5]) + distance(points[2], points[4])) / 2
    return width, height

def detect_eye_state(image):
    h, w = image.shape[:2]
    result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    left_w, left_h = get_eye_size(LEFT_EYE, landmarks, w, h)
    right_w, right_h = get_eye_size(RIGHT_EYE, landmarks, w, h)
    left_ratio = left_h / left_w if left_w != 0 else 0
    right_ratio = right_h / right_w if right_w != 0 else 0
    is_blinking = left_ratio < 0.2 and right_ratio < 0.2
    return {
        "left_w": round(left_w, 1), "left_h": round(left_h, 1),
        "right_w": round(right_w, 1), "right_h": round(right_h, 1),
        "blink": int(is_blinking),
        "eye_state": "Closed" if is_blinking else "Open"
    }

# ---------------- Face Details ----------------
def gender_from_confidence(gender_scores):
    if not isinstance(gender_scores, dict):
        g = str(gender_scores).lower()
        if 'man' in g:
            return "male"
        if 'woman' in g or 'female' in g:
            return "female"
        return "Unknown"

    scores = {k.lower(): v for k, v in gender_scores.items()}
    man_conf = scores.get("man", 0)
    woman_conf = scores.get("woman", 0)
    margin = abs(man_conf - woman_conf)
    if margin < 25:
        return "Unknown"
    return "male" if man_conf > woman_conf else "female"

def interpret_emotion(emotion_scores):
    if not isinstance(emotion_scores, dict):
        return "unknown", "Unknown (0.0%)", 0.0
    main = max(emotion_scores, key=emotion_scores.get)
    pct = emotion_scores[main]
    return main.lower(), f"{main.capitalize()} ({pct:.1f}%)", float(pct)

def estimate_age(age):
    age = age - 3 + random.uniform(-1.5, 1.5)
    return max(16, min(50, age))

def age_to_range(age):
    center = round(age)
    return f"{center - 3}–{center + 3}"

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

# ---------------- HTML Report ----------------
def generate_html(face_info, eye_info):
    img_rows = ""
    for i, f in enumerate(face_info, 1):
        img_rows += f'<img src="data:image/jpeg;base64,{f["base64"]}" width="80" class="rounded-lg shadow-md">\n'
    
    face_rows = ""
    for i, f in enumerate(face_info, 1):
        face_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{f.get('age','N/A')}</td>
            <td>{f.get('gender','N/A')}</td>
            <td>{f.get('emotion','N/A')}</td>
            <td><audio controls><source src="data:audio/mpeg;base64,{f.get('audio_base64','')}"></audio></td>
            <td><span class="status status-{f.get('emotion_key', 'unknown')}"></span></td>
        </tr>
        """
        
    eye_rows = ""
    for idx, e in enumerate(eye_info, 1):
        eye_rows += f"""
        <tr>
            <td>{idx}</td>
            <td>{e.get('left_w','N/A')}×{e.get('left_h','N/A')}</td>
            <td>{e.get('right_w','N/A')}×{e.get('right_h','N/A')}</td>
            <td>{e.get('eye_state','N/A')}</td>
            <td>{e.get('blink',0)}</td>
        </tr>
        """
        
    labels = [f"Face {i+1}" for i in range(len(face_info))]
    ages = [float(f.get('age_numeric', 16)) for f in face_info]
    confidences = [float(f.get('confidence_numeric', 0)) for f in face_info]
    emotions_scores = [float(f.get('emotion_score', 0)) for f in face_info]

    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Facial Analysis Report</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f3f4f6;
                color: #1f2937;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{
                width: 100%;
                max-width: 1200px;
                padding: 2rem;
                display: flex;
                flex-direction: column;
                gap: 2rem;
            }}
            h1, h2, h3, h4 {{
                color: #111827;
                font-weight: 600;
            }}
            .header {{
                text-align: center;
                margin-bottom: 2rem;
            }}
            .card {{
                background-color: #ffffff;
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }}
            .flex-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
                justify-content: center;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
                background-color: #ffffff;
                border-radius: 0.5rem;
                overflow: hidden;
            }}
            th, td {{
                padding: 0.75rem 1rem;
                text-align: left;
                border-bottom: 1px solid #e5e7eb;
            }}
            th {{
                background-color: #f9fafb;
                font-weight: 600;
            }}
            .status {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: #9ca3af;
                position: relative;
                top: 2px;
            }}
            .status-happy {{ background-color: #22c55e; }}
            .status-sad {{ background-color: #3b82f6; }}
            .status-angry {{ background-color: #ef4444; }}
            .status-neutral {{ background-color: #6b7280; }}
            @media (max-width: 768px) {{
                .container {{ padding: 1rem; }}
                table, thead, tbody, th, td, tr {{
                    display: block;
                }}
                thead tr {{
                    position: absolute;
                    top: -9999px;
                    left: -9999px;
                }}
                tr {{ border: 1px solid #ccc; margin-bottom: 0.5rem; }}
                td {{
                    border: none;
                    border-bottom: 1px solid #eee;
                    position: relative;
                    padding-left: 50%;
                    text-align: right;
                }}
                td:before {{
                    content: attr(data-label);
                    position: absolute;
                    left: 0;
                    width: 50%;
                    padding-left: 1rem;
                    white-space: nowrap;
                    font-weight: bold;
                    text-align: left;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="text-3xl font-bold">Facial Analysis Report</h1>
                <p class="text-gray-500">Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="card">
                <h2>Detected Faces</h2>
                <p>A gallery of all the faces captured and analyzed.</p>
                <div class="flex-container my-4">
                    {img_rows}
                </div>
            </div>

            <div class="card">
                <h2>Facial Attributes</h2>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Age</th>
                            <th>Gender</th>
                            <th>Emotion</th>
                            <th>Listen</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {face_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>Eye State & Blinking</h2>
                <p>Metrics on eye openness and blinking frequency.</p>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Left Eye (W×H)</th>
                            <th>Right Eye (W×H)</th>
                            <th>Eye State</th>
                            <th>Blink Count</th>
                        </tr>
                    </thead>
                    <tbody>
                        {eye_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>Combined Analysis</h2>
                <p>Visual representation of captured data. </p>
                <canvas id="combinedChart"></canvas>
            </div>
        </div>

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
                            backgroundColor: '#2563eb'
                        }},
                        {{
                            label: 'Confidence %',
                            data: confidences,
                            backgroundColor: '#16a34a'
                        }},
                        {{
                            label: 'Emotion Score %',
                            data: emotions,
                            backgroundColor: '#f97316'
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
    </body>
    </html>
    '''
    with open("face_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("📄 Results saved to face_report.html")

# ---------------- Main Loop ----------------
def main():
    cap = cv2.VideoCapture(PHONE_CAMERA_URL)
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
                
                # Expand bounding box for better face analysis
                padding = int(w * 0.25)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(iw, x + w + padding)
                y2 = min(ih, y + h + padding)
                face_img = frame[y1:y2, x1:x2]

                face_path = f"faces/face_{counter}.jpg"
                cv2.imwrite(face_path, face_img)

                try:
                    dfa = DeepFace.analyze(
                        face_img,
                        actions=["gender", "emotion", "age"],
                        enforce_detection=False,
                        detector_backend="mtcnn"
                    )
                    result = dfa[0] if isinstance(dfa, list) and dfa else dfa
                except Exception as e:
                    print(f"DeepFace analyze error: {e}")
                    result = {"gender": "unknown", "emotion": {}, "age": 25}

                gender = gender_from_confidence(result.get("gender", {}))
                emotion_key, emotion_text, emotion_score = interpret_emotion(result.get("emotion", {}))
                age_raw = result.get("age", 25)
                age_range = age_to_range(estimate_age(age_raw))
                age_numeric = round(age_raw, 2)
                confidence = round(det.score[0] * 100, 2)
                
                with open(face_path, "rb") as img_file:
                    base64_img = base64.b64encode(img_file.read()).decode()

                text = (f"This person is {emotion_text}, gender = {gender}, age between {age_range} years.")
                audio_path = f"audios/audio_{counter}.mp3"
                make_audio(text, audio_path)
                with open(audio_path, "rb") as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode()
                
                music_file = EMOTION_MUSIC.get(emotion_key, EMOTION_MUSIC["neutral"])
                src_music_path = os.path.join("music_library", music_file)
                music_output_path = f"audios/music_{counter}.mp3"
                if os.path.exists(src_music_path):
                    shutil.copy(src_music_path, music_output_path)
                
                eye_result = detect_eye_state(face_img) or {
                    "left_w": "N/A", "left_h": "N/A",
                    "right_w": "N/A", "right_h": "N/A",
                    "blink": 0, "eye_state": "Unknown"
                }

                faces_data.append({
                    "base64": base64_img,
                    "gender": gender,
                    "emotion_key": emotion_key,
                    "emotion": emotion_text,
                    "emotion_score": emotion_score,
                    "age": age_range,
                    "age_numeric": age_numeric,
                    "confidence_numeric": confidence,
                    "audio_base64": audio_base64
                })

                eyes_data.append(eye_result)
                print(f"✅ Saved Face {counter} | gender = {gender}, {emotion_text}, Age: {age_range}, Confidence: {confidence}%")
                counter += 1
                
                generate_html(faces_data, eyes_data)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
