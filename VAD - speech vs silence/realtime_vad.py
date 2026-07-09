import time
import numpy as np
import sounddevice as sd
import joblib

from feature_extraction import extract_features

# ===========================
# LOAD MODEL
# ===========================
model = joblib.load("models/vad_model.pkl")

# ===========================
# SETTINGS
# ===========================
SR = 8000
FRAME_DURATION = 0.02          # 20 ms
FRAME_SIZE = int(SR * FRAME_DURATION)

current_state = None

speech_frames = 0
silence_frames = 0

# --- Added Tuneable Settings ---
ENERGY_THRESHOLD = 0.01        # If mic is quieter than this, force SILENCE. 
SMOOTHING_FRAMES = 3          # Requires 3 consecutive speech frames to trigger SPEECH.
consecutive_speech = 0         # Tracks consecutive speech frames

# ===========================
# CALLBACK
# ===========================
def audio_callback(indata, frames, time_info, status):

    global current_state
    global speech_frames, silence_frames
    global consecutive_speech

    frame = indata[:, 0]

    # Measure microphone RMS
    rms = np.sqrt(np.mean(frame**2))
    print(f"RMS: {rms:.4f}")

    # Continue with your existing code
    if rms < ENERGY_THRESHOLD:
        pred = 0
        prob = [1.0, 0.0]
    else:
        features = extract_features(frame, SR)
        X = np.array(features).reshape(1, -1)

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        speech_probability = prob[1]

        if speech_probability > 0.85:
            pred = 1
        else:
            pred = 0
        prob = model.predict_proba(X)[0]
        
    print(f"RMS: {rms:.4f} | Prob: {prob}")

    # Debounce / Smoothing Logic
    if pred == 1:
        consecutive_speech += 1
    else:
        consecutive_speech = 0

    # Determine final decision based on history
    if consecutive_speech >= SMOOTHING_FRAMES:
        speech_frames += 1
        new_state = "SPEECH"
        confidence = prob[1] * 100
    else:
        silence_frames += 1
        new_state = "SILENCE"
        confidence = prob[0] * 100

    # Print ONLY when state changes
    if new_state != current_state:
        print(f"\n[{time.strftime('%H:%M:%S')}] {new_state} ({confidence:.1f}%)")
        current_state = new_state


# ===========================
# START
# ===========================
print("Listening for 30 seconds...")
print("Speak normally. Stay silent to test silence detection.\n")

with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=FRAME_SIZE,
        callback=audio_callback):

    sd.sleep(30000)      # 30 seconds

# ===========================
# SUMMARY
# ===========================
total = speech_frames + silence_frames

print("\n========== TEST SUMMARY ==========")
print("Total Frames   :", total)
print("Speech Frames  :", speech_frames)
print("Silence Frames :", silence_frames)

if total > 0:
    speech_percent = (speech_frames / total) * 100
    silence_percent = (silence_frames / total) * 100

    print(f"Speech  : {speech_percent:.2f}%")
    print(f"Silence : {silence_percent:.2f}%")

print("==================================")