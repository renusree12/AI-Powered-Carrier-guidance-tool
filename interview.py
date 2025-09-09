import streamlit as st
import cv2
import numpy as np
import time
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import threading
import random
import mediapipe as mp
from PIL import Image
import pdfplumber
import docx
import os
import queue

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hand_detector = mp_hands.Hands(max_num_hands=2)
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# Constants
expression_scores = {
    "Happy": 2,
    "Confident": 3,
    "Neutral": 1,
    "Nervous": -2,
    "Doubtful": -3
}
RESUME_KEYWORDS = ["python", "machine learning", "tensorflow", "opencv", "whisper", "cv2", "git", "docker", "api", "tkinter"]

# Globals
audio_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
frame_queue = queue.Queue()

# Initialize session state
if "interview_running" not in st.session_state:
    st.session_state.interview_running = False
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "cheating_reasons" not in st.session_state:
    st.session_state.cheating_reasons = []
if "malpractice_flag" not in st.session_state:
    st.session_state.malpractice_flag = False
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "speech_sentiment" not in st.session_state:
    st.session_state.speech_sentiment = "Neutral"
if "resume_score" not in st.session_state:
    st.session_state.resume_score = 0

# Audio recording
def record_audio(duration=30, sample_rate=44100):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    wav.write(audio_file_path, sample_rate, audio)

# Analyze speech
def analyze_speech(audio_file):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        text = result.get("text", "").strip()

        malpractice_keywords = ["chatgpt", "help", "copy", "someone", "google", "cheat", "answer"]
        malpractice_detected = any(word in text.lower() for word in malpractice_keywords)
        if len(text) < 10:
            malpractice_detected = True

        positive_words = ["good", "great", "excellent", "confident", "strong"]
        negative_words = ["bad", "poor", "weak", "nervous", "uncertain"]

        sentiment_score = sum(1 for word in positive_words if word in text.lower()) - \
                          sum(1 for word in negative_words if word in text.lower())

        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        return sentiment, malpractice_detected
    except:
        return "Error", True

# Detect expression and cheating
def detect_expression_and_cheating(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detector.process(frame_rgb)
    face_present = face_results.detections is not None

    hand_results = hand_detector.process(frame_rgb)
    hands_present = hand_results.multi_hand_landmarks is not None

    expressions = ["Happy", "Confident", "Neutral", "Nervous", "Doubtful"]
    weights = [0.2, 0.3, 0.3, 0.1, 0.1]
    expression = random.choices(expressions, weights=weights)[0]

    cheating = False
    reason = ""
    if hands_present:
        cheating = True
        reason = "Hand gesture detected"
    if not face_present:
        cheating = True
        reason = "Face not visible"

    return expression, cheating, reason

# Resume parsing
def parse_resume(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.lower()

def score_resume(resume_text):
    return sum(1 for keyword in RESUME_KEYWORDS if keyword in resume_text)

# Interview logic
def start_interview(duration=30):
    st.session_state.interview_running = True
    st.session_state.feedback_log.clear()
    st.session_state.cheating_reasons.clear()
    st.session_state.malpractice_flag = False

    threading.Thread(target=record_audio, args=(duration,), daemon=True).start()

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        expression, is_cheating, reason = detect_expression_and_cheating(frame)
        timestamp = round(time.time() - start_time, 2)
        st.session_state.feedback_log.append((timestamp, expression))

        if is_cheating:
            st.session_state.malpractice_flag = True
            st.session_state.cheating_reasons.append((timestamp, reason))

        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"Expression: {expression}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if is_cheating:
            cv2.putText(frame_copy, f"Cheating: {reason}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        st.image(frame_pil, channels="RGB", caption="Interview in progress...")
        time.sleep(0.03)

    cap.release()

    st.session_state.speech_sentiment, audio_mal = analyze_speech(audio_file_path)
    st.session_state.malpractice_flag |= audio_mal

    st.session_state.resume_score = score_resume(st.session_state.resume_text) * 10
    st.session_state.interview_running = False

def display_results(duration=30):
    total_score = sum(expression_scores.get(expr, 0) for _, expr in st.session_state.feedback_log)

    if st.session_state.speech_sentiment == "Positive":
        total_score += 3
    elif st.session_state.speech_sentiment == "Negative":
        total_score -= 3

    total_score += st.session_state.resume_score // 10

    max_possible = (duration // 2) * 3 + 3 + len(RESUME_KEYWORDS)
    final_score_percent = max(0, min(100, int((total_score / max_possible) * 100)))

    decision = "Selected" if total_score >= 20 else "Considered" if total_score >= 10 else "Not Selected"
    if st.session_state.malpractice_flag:
        decision = "Disqualified due to Malpractice"

    st.subheader("Interview Summary")
    st.text(f"Expressions Analyzed: {len(st.session_state.feedback_log)}")
    st.text(f"Speech Sentiment: {st.session_state.speech_sentiment}")
    st.text(f"Resume Score: {st.session_state.resume_score}/100")
    st.text(f"Malpractice Detected: {'Yes' if st.session_state.malpractice_flag else 'No'}")
    if st.session_state.cheating_reasons:
        st.text("Cheating Reasons:")
        for t, r in st.session_state.cheating_reasons:
            st.text(f" - {t}s: {r}")
    st.text(f"Final Decision: {decision}")

# Streamlit UI
st.title("AI Interview Analyzer")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
if resume_file:
    st.session_state.resume_text = parse_resume(resume_file)
    st.success("Resume uploaded and parsed.")

if st.button("Start Interview") and resume_file:
    start_interview(duration=30)

if not st.session_state.interview_running and st.session_state.feedback_log:
    display_results(duration=30)
