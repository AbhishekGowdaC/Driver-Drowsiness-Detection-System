import cv2
import dlib
import time
import math
import pygame
import numpy as np
import csv
import os
import pyttsx3
from threading import Thread
from collections import deque

# ------------------ INITIAL SETUP ------------------
pygame.init()
pygame.mixer.init()

ALARM_SOUND = "alarm.mp3"
TICK_SOUND  = "tick.mp3"

try:
    tick_sound = pygame.mixer.Sound(TICK_SOUND)
except Exception:
    tick_sound = None

# Initialize joystick/gamepad support
pygame.joystick.init()
joysticks = []
for i in range(pygame.joystick.get_count()):
    js = pygame.joystick.Joystick(i)
    js.init()
    joysticks.append(js)
    print(f"🎮 Gamepad detected: {js.get_name()}")

# Voice alert engine (runs in background thread to avoid blocking)
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)
tts_busy = False

def speak(text):
    global tts_busy
    if tts_busy:
        return
    def _run():
        global tts_busy
        tts_busy = True
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_busy = False
    Thread(target=_run, daemon=True).start()

speed              = 80
alarm_on           = False
hazard_on          = False
car_on             = True
last_drowsy_time   = 0
last_voice_time    = 0
face_drowsy_start  = 0
blink_count        = 0
blink_minute_start = time.time()
eye_was_closed     = False

live = dict(ear=0.0, mar=0.0, tilt=0.0, perclos=0.0, yaw=0.0,
            risk_score=0, blink_rate=0, steer_conf=1.0)

# ------------------ PERCLOS ------------------
eye_state_history = deque(maxlen=60)
PERCLOS_THRESH    = 0.35

# ------------------ LOGGING ------------------
LOG_FILE = "drowsiness_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["timestamp", "ear", "mar", "perclos", "yaw",
             "blink_rate", "risk_score", "steer_conf", "speed", "risk_level"])

def log_frame(risk_level):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{live['ear']:.3f}", f"{live['mar']:.3f}",
            f"{live['perclos']:.3f}", f"{live['yaw']:.1f}",
            live['blink_rate'], live['risk_score'],
            f"{live['steer_conf']:.2f}", speed, risk_level
        ])

# ------------------ MODELS ------------------
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EYE_AR_THRESH      = 0.23
MOUTH_AR_THRESH    = 0.60   # closed mouth ~0.2-0.3, yawn/open ~0.6+
HEAD_TILT_THRESH   = 15
YAW_THRESH         = 25
SPEED_DEC_INTERVAL = 10
DROWSY_CONFIRM_SEC = 3

# 3D model reference points for head pose
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0, -65.0),
    (-225.0, 170.0,-135.0),
    (225.0,  170.0,-135.0),
    (-150.0,-150.0,-125.0),
    (150.0, -150.0,-125.0),
], dtype=np.float64)

# ------------------ AUDIO ------------------
def play_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        try:
            pygame.mixer.music.load(ALARM_SOUND)
            pygame.mixer.music.play(-1)
        except Exception:
            pass

def stop_alarm():
    global alarm_on
    if alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False

def play_tick():
    if tick_sound:
        tick_sound.play()

# ------------------ FEATURE CALCULATIONS ------------------
def eye_aspect_ratio(eye):
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def mouth_aspect_ratio(mouth):
    # Average of two vertical distances divided by horizontal width
    # Closed mouth ~ 0.2-0.3, open/yawning mouth ~ 0.6+
    A = math.dist(mouth[2], mouth[10])   # vertical inner top-bottom
    B = math.dist(mouth[4], mouth[8])    # vertical inner top-bottom
    C = math.dist(mouth[0], mouth[6])    # horizontal corner to corner
    return ((A + B) / 2.0) / C if C > 0 else 0.0

def get_head_tilt(pts):
    dx = pts[45][0] - pts[36][0]
    dy = pts[45][1] - pts[36][1]
    return math.degrees(math.atan2(dy, dx))

def compute_perclos():
    if len(eye_state_history) < 10:
        return 0.0
    return sum(1 for s in eye_state_history if not s) / len(eye_state_history)

def get_yaw(pts, frame_shape):
    h, w = frame_shape[:2]
    focal  = w
    center = (w / 2, h / 2)
    cam_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    image_points = np.array([
        pts[30], pts[8], pts[36], pts[45], pts[48], pts[54]
    ], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points,
                                cam_matrix, dist_coeffs,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    yaw = math.degrees(math.atan2(rmat[1][0], rmat[0][0]))
    return yaw

def update_blink_rate(ear):
    global blink_count, blink_minute_start, eye_was_closed
    eye_closed_now = ear < EYE_AR_THRESH
    if eye_was_closed and not eye_closed_now:
        blink_count += 1
    eye_was_closed = eye_closed_now
    elapsed = time.time() - blink_minute_start
    if elapsed >= 60:
        live['blink_rate'] = blink_count
        blink_count        = 0
        blink_minute_start = time.time()
    elif elapsed > 5:
        live['blink_rate'] = int(blink_count * (60 / elapsed))

# ============================================================
# IMPROVED STEERING ACTIVITY SYSTEM
# ============================================================

STEERING_TIMEOUT   = 8.0    # seconds until confidence reaches 0

last_steering_time  = time.time()
_quit_requested     = False
_alarm_dismissed    = False


def steering_confidence():
    """
    Returns a float 0.0–1.0.
    1.0 = steering just happened, 0.0 = fully idle for STEERING_TIMEOUT seconds.
    Decays linearly over the timeout window.
    """
    elapsed = time.time() - last_steering_time
    return max(0.0, 1.0 - (elapsed / STEERING_TIMEOUT))


def steering_inactive():
    """Binary flag: True when confidence drops below 10%."""
    return steering_confidence() < 0.10


def _register_steering():
    """Record a valid steering event."""
    global last_steering_time
    last_steering_time = time.time()


def process_steering_inputs(key):
    """
    Call once per frame with the key returned by cv2.waitKey().
    Uses the OpenCV window for input — always has focus, always reliable.
    Returns: (quit_requested, alarm_dismissed)
    """
    global _quit_requested, _alarm_dismissed

    _quit_requested  = False
    _alarm_dismissed = False

    if key == -1:                        # no key pressed this frame
        return False, False

    if key in (ord('a'), ord('d')):
        _register_steering()

    elif key == ord('q'):
        _quit_requested = True

    elif key in (13, 32):                # ENTER or SPACE
        _alarm_dismissed = True
        _register_steering()

    return _quit_requested, _alarm_dismissed

# ============================================================
# END STEERING SYSTEM
# ============================================================

# ------------------ RISK SCORE MODEL ------------------
def compute_risk_score(ear, mar, perclos, tilt, yaw, steer_conf):
    """
    Weighted risk scoring with graded steering confidence penalty.
    Max possible score = 13.
    """
    score = 0

    if ear     < EYE_AR_THRESH:        score += 2
    if mar     > MOUTH_AR_THRESH:      score += 1
    if perclos > PERCLOS_THRESH:       score += 3
    if abs(tilt) > HEAD_TILT_THRESH:   score += 1
    if abs(yaw)  > YAW_THRESH:         score += 2

    # Graded steering penalty (was binary +2 / 0)
    if steer_conf < 0.10:              score += 2   # fully idle
    elif steer_conf < 0.50:            score += 1   # fading

    # Blink rate anomaly
    br = live['blink_rate']
    if 0 < br < 8 or br > 25:          score += 2

    return score


def evaluate_risk(score):
    if score >= 6:
        return "HIGH"
    elif score >= 3:
        return "MEDIUM"
    else:
        return "SAFE"

# ------------------ FACE DETECTION & METRICS ------------------
def detect_face(frame, gray):
    global face_drowsy_start

    faces = detector(gray)

    if len(faces) == 0:
        eye_state_history.append(True)
        face_drowsy_start = 0
        cv2.putText(frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return 0.0, 0.0, 0.0, 0.0, 0.0

    face  = faces[0]
    shape = predictor(gray, face)
    pts   = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for (x, y) in pts:
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    for region, color in [(pts[36:42], (0, 200, 255)),
                          (pts[42:48], (0, 200, 255)),
                          (pts[48:68], (0, 255, 128))]:
        hull = cv2.convexHull(np.array(region))
        cv2.drawContours(frame, [hull], -1, color, 1)

    left_eye  = pts[36:42]
    right_eye = pts[42:48]
    mouth     = pts[48:68]

    ear     = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    mar     = mouth_aspect_ratio(mouth)
    tilt    = get_head_tilt(pts)
    yaw     = get_yaw(pts, frame.shape)

    eye_state_history.append(ear >= EYE_AR_THRESH)
    perclos = compute_perclos()

    update_blink_rate(ear)
    live.update(ear=ear, mar=mar, tilt=tilt, perclos=perclos, yaw=yaw)

    face_alert = (ear < EYE_AR_THRESH) or (perclos > PERCLOS_THRESH)
    if face_alert:
        if face_drowsy_start == 0:
            face_drowsy_start = time.time()
        elapsed = time.time() - face_drowsy_start
        pct   = min(elapsed / DROWSY_CONFIRM_SEC, 1.0)
        bar_w = int(250 * pct)
        h     = frame.shape[0]
        cv2.rectangle(frame, (10, h - 25), (260, h - 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, h - 25), (10 + bar_w, h - 10), (0, 80, 255), -1)
        cv2.putText(frame, f"Drowsy timer {elapsed:.1f}s", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
    else:
        if face_drowsy_start > 0 and (time.time() - face_drowsy_start) < 1.0:
            pass
        else:
            face_drowsy_start = 0

    return ear, mar, tilt, yaw, perclos

# ------------------ SPEED CONTROL ------------------
def decrease_speed():
    global speed, hazard_on, car_on
    if speed > 0:
        speed = max(0, speed - 10)
        hazard_on = True
        play_tick()
        print(f"⚠  Speed reduced to {speed} km/h")
    if speed <= 0:
        car_on = False
        print("🛑 Car stopped — no driver response")

# ------------------ HUD & OVERLAYS ------------------
def draw_steering_confidence_bar(frame, conf):
    """
    Draws a small horizontal confidence bar beneath the HUD.
    Green → Yellow → Red as confidence drops.
    """
    h, w = frame.shape[:2]
    bar_x, bar_y, bar_w, bar_h = 12, h - 58, 230, 8
    filled = int(bar_w * conf)

    # Background
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)

    # Colour: green above 50%, orange 20–50%, red below 20%
    if conf > 0.50:
        bar_color = (0, 200, 60)
    elif conf > 0.20:
        bar_color = (0, 140, 255)
    else:
        bar_color = (30, 30, 220)

    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), bar_color, -1)

    cv2.putText(frame, f"Steer conf {conf:.0%}",
                (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_color, 1)


def draw_hud(frame, steer_conf):
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, h - 205), (255, h - 35), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def row(label, val_str, y, ok):
        color = (0, 210, 80) if ok else (30, 60, 230)
        cv2.putText(frame, f"{label:<11}{val_str}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

    y0 = h - 195
    row("EAR",       f"{live['ear']:.3f}",        y0,     live['ear']       >= EYE_AR_THRESH)
    row("MAR",       f"{live['mar']:.3f}",        y0+20,  live['mar']       <= MOUTH_AR_THRESH)
    row("Tilt",      f"{live['tilt']:.1f}°",      y0+40,  abs(live['tilt']) <= HEAD_TILT_THRESH)
    row("PERCLOS",   f"{live['perclos']:.0%}",    y0+60,  live['perclos']   <= PERCLOS_THRESH)
    row("Yaw",       f"{live['yaw']:.1f}°",       y0+80,  abs(live['yaw'])  <= YAW_THRESH)
    row("Blink/min", f"{live['blink_rate']}",     y0+100, 8 <= live['blink_rate'] <= 25)
    row("Risk",      f"{live['risk_score']}/13",  y0+120, live['risk_score'] < 3)

    # Steering confidence row with % indicator
    steer_ok = steer_conf >= 0.50
    row("Steering",  f"{steer_conf:.0%}",         y0+140, steer_ok)

    draw_steering_confidence_bar(frame, steer_conf)


def draw_status(frame, risk):
    configs = {
        "HIGH":   ("DROWSY  !",  (0,   0, 220)),
        "MEDIUM": ("WARNING  ~", (0, 140, 255)),
        "SAFE":   ("ACTIVE  OK", (0, 200,  60)),
    }
    label, color = configs[risk]
    cv2.putText(frame, label, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def draw_warnings(frame, steering_idle, yaw):
    y = 70
    if abs(yaw) > YAW_THRESH:
        direction = "LEFT" if yaw < 0 else "RIGHT"
        cv2.putText(frame, f"! Distracted — looking {direction}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 1)
        y += 24
    if steering_idle:
        cv2.putText(frame, "! No steering input", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 1)


def draw_hazard(frame):
    if hazard_on:
        cv2.circle(frame, (60, 60), 28, (0, 145, 255), -1)
        cv2.putText(frame, "HAZARD", (22, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 145, 255), 2)


def draw_controls_hint(frame):
    """Show control hints in the top-right corner."""
    w = frame.shape[1]
    hints = [
        "A / D        : steer",
        "ENTER/SPACE  : dismiss",
        "Q            : quit",
    ]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (w - 230, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

# ------------------ MAIN LOOP ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found")
    exit(1)

# Create a tiny pygame window (hidden behind the OpenCV window)
# so pygame can receive events.  Size 1x1 — we only need the event pump.
os.environ.setdefault('SDL_VIDEO_WINDOW_POS', '0,0')
pygame.display.set_mode((1, 1), pygame.NOFRAME)
pygame.display.set_caption("DMS Event Receiver")

print("🔵 Driver Monitoring System Running")
print("   Steering : A / D keys (click the OpenCV window first!)")
print("   ENTER / SPACE : dismiss alarm   |  Q : quit\n")

log_interval = 0

try:
    while car_on:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame capture failed")
            break

        # ── Key input via OpenCV window (always has focus) ───────
        key = cv2.waitKey(1) & 0xFF
        quit_req, alarm_dismissed = process_steering_inputs(key)

        if quit_req:
            break

        # ── Vision processing ─────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        ear, mar, tilt, yaw, perclos = detect_face(frame, gray)

        # ── Steering confidence ───────────────────────────────────
        steer_conf    = steering_confidence()
        steering_idle = steer_conf < 0.10
        live['steer_conf'] = steer_conf

        # ── Risk scoring ──────────────────────────────────────────
        score = compute_risk_score(ear, mar, perclos, tilt, yaw, steer_conf)
        live['risk_score'] = score
        risk  = evaluate_risk(score)

        # ── Logging ───────────────────────────────────────────────
        log_interval = (log_interval + 1) % 30
        if log_interval == 0:
            log_frame(risk)

        # ── Alerts & actions ──────────────────────────────────────
        if risk == "HIGH":
            play_alarm()
            now = time.time()
            if now - last_drowsy_time > SPEED_DEC_INTERVAL:
                decrease_speed()
                last_drowsy_time = now
            if now - last_voice_time > 10:
                speak("Warning! Driver is drowsy. Please take a break.")
                last_voice_time = now
            if alarm_dismissed:
                print("✅ Driver confirmed active")
                stop_alarm()
                hazard_on         = False
                face_drowsy_start = 0
                eye_state_history.clear()

        elif risk == "MEDIUM":
            stop_alarm()
            hazard_on = False
        else:
            stop_alarm()
            hazard_on = False

        # ── Draw overlays ─────────────────────────────────────────
        draw_status(frame, risk)
        draw_warnings(frame, steering_idle, yaw)
        draw_hud(frame, steer_conf)
        draw_hazard(frame)
        draw_controls_hint(frame)

        cv2.putText(frame, f"Speed: {speed} km/h",
                    (frame.shape[1] - 210, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)

        time.sleep(0.03)

except KeyboardInterrupt:
    print("\n[Stopped by user]")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print(f"✅ Program ended. Log saved to: {LOG_FILE}")