# app.py
import cv2
import numpy as np
#from ses_metin import listen_and_write_segment
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import threading
import time
import torch
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import tempfile
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from flask import Flask, render_template, Response, jsonify
import queue
import logging
import os
import speech_recognition as sr
# Eklenecek yeni importlar
from flask import send_from_directory  # Yeni eklenen
from flask_cors import CORS  # Yeni eklenen

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


# Konsol çıktılarını yakalamak için kuyruk
console_queue = queue.Queue()

# Global değişkenler
match_count = 0
audio_emotions = []
stop_analysis = False
analysis_active = False
speech_queue = queue.Queue()


# Yüz analizi için model ve etiketler
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ses analizi için model ve yapılandırma
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name_or_path).to(device)

# Eşanlamlı duygu eşleştirme
emotion_mapping = {
    "Sad": "Sadness",
    "Angry": "Anger",
    "Happy": "Neutral",
    "Happiness": "Neutral",
    "Neutral": "Neutral",
    "Surprise": "Surprise",
}

def log_to_queue(message):
    """Konsol mesajlarını kuyruğa ekler"""
    timestamp = time.strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    console_queue.put(formatted_message)
    print(formatted_message)

def map_emotion(emotion):
    """Eşanlamlı duyguları dönüştüren fonksiyon."""
    return emotion_mapping.get(emotion, emotion)

def predict_face(frame):
    """Yüz analizini gerçekleştiren fonksiyon."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    face_emotions = []
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            face_emotions.append(label)
            
            # Yüz çerçevesi ve duygu etiketi çizme
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return face_emotions, frame

def predict_audio(audio_data, sampling_rate):
    """Ses analizini gerçekleştiren fonksiyon."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file_path = temp_file.name
        torchaudio.save(temp_file_path, torch.tensor(audio_data.T), sampling_rate)

    speech = torchaudio.load(temp_file_path)[0].squeeze().numpy()
    inputs = feature_extractor(speech, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": scores[i]} for i in range(len(scores))]
    
    os.unlink(temp_file_path)
    return sorted(outputs, key=lambda x: x["Score"], reverse=True)

def audio_analysis():
    """Ses analizi her 5 saniyede bir gerçekleştirilir."""
    global audio_emotions, stop_analysis
    while not stop_analysis:
        if analysis_active:
            try:
                audio_data = sd.rec(int(5 * sampling_rate), samplerate=sampling_rate, channels=1)
                sd.wait()
                audio_emotions = predict_audio(audio_data, sampling_rate)
                log_to_queue(f"Ses duyguları: {audio_emotions[:2]}")
            except Exception as e:
                log_to_queue(f"Ses analizi hatası: {str(e)}")
        time.sleep(0.1)

def generate_frames():
    """Video akışını sağlayan generator fonksiyon"""
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if analysis_active:
            face_emotions, processed_frame = predict_face(frame)
            # Son duyguları sakla
            generate_frames.last_emotions = face_emotions
            frame = processed_frame
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Son duyguları saklamak için statik değişken
generate_frames.last_emotions = []
# Global değişkenlere speech_active ekleyelim
speech_active = False

def speech_to_text_thread(output_file, total_duration):
    """
    Speech to text thread - artık speech_active kontrolü ile çalışacak
    """
    global stop_analysis, analysis_active, speech_active
    
    # Speech_active False ise hiç başlama
    if not speech_active:
        return
        
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Ses tanıma ayarları
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.5
    
    current_segment = 1
    segment_texts = {i: [] for i in range(1, 13)}
    loop_duration = 24
    
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Analiz Başladı:\n\n")
    
    with microphone as source:
        log_to_queue("Ortam sesi ayarlanıyor...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        log_to_queue("Konuşma dinleme başladı...")
        log_to_queue("Döngü 1 için konuşma dinlemeye başlandı")
        
        start_time = time.time()
        last_segment_time = start_time
        
        while speech_active and current_segment <= 12:  # speech_active kontrolü eklendi
            try:
                current_time = time.time()
                elapsed_segment_time = current_time - last_segment_time
                
                if elapsed_segment_time >= loop_duration:
                    text_to_write = " ".join(segment_texts[current_segment])
                    with open(output_file, "a", encoding="utf-8") as file:
                        if text_to_write:
                            file.write(f"[DÖNGÜ {current_segment}]:\n{text_to_write}\n\n")
                        else:
                            file.write(f"[DÖNGÜ {current_segment}]: [Konuşma algılanamadı]\n\n")
                    
                    current_segment += 1
                    if current_segment <= 12:
                        last_segment_time = current_time
                        log_to_queue(f"Döngü {current_segment} için konuşma dinlemeye başlandı")
                
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
                
                try:
                    text = recognizer.recognize_google(audio, language="tr-TR")
                    if text and current_segment <= 12:
                        log_to_queue(f"[Döngü {current_segment}] Algılanan metin: {text}")
                        segment_texts[current_segment].append(text)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    log_to_queue("Google API hatası")
                    continue
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                log_to_queue(f"Hata: {str(e)}")
                continue
    
    if current_segment <= 12 and segment_texts[current_segment]:
        text_to_write = " ".join(segment_texts[current_segment])
        with open(output_file, "a", encoding="utf-8") as file:
            if text_to_write:
                file.write(f"[DÖNGÜ {current_segment}]:\n{text_to_write}\n\n")
    
    log_to_queue("Konuşma dinleme tamamlandı")
    speech_active = False
def video_analysis(round_count):
    """Video analizini yöneten fonksiyon."""
    global match_count, stop_analysis, analysis_active, speech_active
    
    stress_score = 0
    match_count = 0
    results = None
    
    for i in range(1, round_count + 1):
        log_to_queue(f"\n=== Döngü {i} başladı ===")
        start_time = time.time()
        last_analysis_time = 0
        analysis_count = 0  # Bu döngüdeki analiz sayısını takip et
        
        # Her döngü 24 saniye
        while time.time() - start_time < 24 and analysis_count < 6:  
            current_time = time.time()
            
            # Her 4 saniyede bir analiz yap
            if current_time - last_analysis_time >= 4:
                analysis_count += 1
                log_to_queue(f"\nAnaliz {analysis_count}/6 başladı...")
                
                # Yüz duygularını al
                face_emotions = []
                if frame_emotions := getattr(generate_frames, 'last_emotions', []):
                    face_emotions = frame_emotions
                    log_to_queue(f"Yüz duygusu: {face_emotions}")
                
                # Ses duygularını kontrol et
                if audio_emotions:
                    top_audio_emotions = audio_emotions[:2]  # En yüksek 2 duygu
                    log_to_queue(f"En yüksek ses duyguları: {[e['Label'] for e in top_audio_emotions]}")
                    
                    # Eşleşme kontrolü
                    if face_emotions:
                        for face_emotion in face_emotions:
                            mapped_face_emotion = map_emotion(face_emotion)
                            audio_labels = [map_emotion(e["Label"]) for e in top_audio_emotions]
                            
                            if mapped_face_emotion in audio_labels:
                                match_count += 1
                                log_to_queue(f"✓ EŞLEŞME BAŞARILI ({analysis_count}/6)!")
                                log_to_queue(f"   Yüz Duygusu: {face_emotion}")
                                log_to_queue(f"   Eşleşen Ses Duygusu: {[e['Label'] for e in top_audio_emotions if map_emotion(e['Label']) == mapped_face_emotion][0]}")
                                
                                # Duygu puanlaması
                                if face_emotion in ["Happy", "Surprise", "Neutral"]:
                                    stress_score += 1
                                elif face_emotion in ["Angry", "Disgust", "Fear", "Sad"]:
                                    stress_score -= 1
                            else:
                                log_to_queue(f"× Eşleşme bulunamadı ({analysis_count}/6)")
                
                last_analysis_time = current_time
            
            time.sleep(0.1)  # CPU yükünü azalt
        
        log_to_queue(f"\n=== Döngü {i} tamamlandı ===")
        log_to_queue(f"Bu döngüde yapılan analiz sayısı: {analysis_count}")
        if analysis_count > 0:
            log_to_queue(f"Bu döngüdeki eşleşme oranı: {match_count/analysis_count:.2%}")
        
        # 12. döngüde sonuçları hesapla ama analizi durdurmadan devam et
        if i == round_count:
            match_bonus = match_count * 1  # Her eşleşme için 1 puan
            stress_score = min(100, max(0, stress_score))  # 0-100 arası sınırlama
            general_analysis_score = min(100, stress_score + match_bonus)
            
            log_to_queue("\n=== GENEL ANALİZ SONUÇLARI ===")
            log_to_queue(f"Toplam Eşleşme Sayısı: {match_count}")
            log_to_queue(f"Duygu Eşleşme Ek Puanı: {match_bonus}")
            log_to_queue(f"Stres Kontrol Puanı: {stress_score}")
            log_to_queue(f"Genel Duygu Analiz Puanı: {general_analysis_score}")
            
            results = {
                "stress_score": stress_score,
                "match_bonus": match_bonus,
                "general_score": general_analysis_score,
                "match_count": match_count
            }
    
    # Sonuçları döndür ama analizi durdurmadan devam et
    while speech_active:
        current_time = time.time()
        
        if current_time - last_analysis_time >= 4:
            # Yüz ve ses analizi devam ediyor
            face_emotions = getattr(generate_frames, 'last_emotions', [])
            if face_emotions and audio_emotions:
                log_to_queue("\nEk analiz devam ediyor...")
                log_to_queue(f"Yüz duygusu: {face_emotions}")
                log_to_queue(f"Ses duyguları: {[e['Label'] for e in audio_emotions[:2]]}")
            
            last_analysis_time = current_time
        
        time.sleep(0.1)
    
    return results
# Flask route'ları
@app.route('/api/analysis_status', methods=['GET'])
def get_analysis_status():
    """Get current analysis status and results"""
    global match_count, audio_emotions, analysis_active
    
    # Get face emotions from the last frame
    face_emotions = getattr(generate_frames, 'last_emotions', [])
    
    # Format audio emotions for response
    current_audio_emotions = []
    if audio_emotions:
        current_audio_emotions = [
            {"emotion": e["Label"], "score": float(e["Score"])} 
            for e in audio_emotions[:2]
        ]
    
    return jsonify({
        "active": analysis_active,
        "match_count": match_count,
        "face_emotions": face_emotions,
        "audio_emotions": current_audio_emotions
    })

@app.route('/api/speech_text', methods=['GET'])
def get_speech_text():
    """Get the latest speech-to-text results"""
    try:
        with open("konusma_metni.txt", "r", encoding="utf-8") as file:
            text = file.read()
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Update the start_analysis route to return more detailed information

@app.route('/api/start_analysis', methods=['GET'])
def start_analysis():
    global analysis_active, stop_analysis, match_count, audio_emotions, speech_active
    
    # Reset all global variables
    analysis_active = True
    stop_analysis = False
    match_count = 0
    audio_emotions = []
    speech_active = True  # Speech to text'i başlat
    
    # Reset the speech text file
    output_file = "konusma_metni.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Analiz Başladı:\n\n")
    
    # Stop existing speech thread if running
    for thread in threading.enumerate():
        if thread.name == "speech_thread":
            stop_analysis = True
            speech_active = False
            thread.join(timeout=1)
            break
    
    # Start new speech thread
    stop_analysis = False
    speech_active = True
    speech_thread = threading.Thread(
        target=speech_to_text_thread,
        args=(output_file, 16),
        name="speech_thread",
        daemon=True
    )
    speech_thread.start()
    
    try:
        results = video_analysis(round_count=12)
        return jsonify({
            "status": "success",
            "results": {
                "stress_score": results["stress_score"],
                "match_bonus": results["match_bonus"],
                "general_score": results["general_score"],
                "match_count": results["match_count"]
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/stop_analysis', methods=['GET'])
def stop_analysis_route():
    global analysis_active, stop_analysis, speech_active
    analysis_active = False
    stop_analysis = True
    speech_active = False  # Speech to text'i durdur
    return jsonify({"status": "success"})


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Access-Control-Allow-Origin': '*'
        }
    )

if __name__ == "__main__":
    audio_thread = threading.Thread(target=audio_analysis, daemon=True)
    audio_thread.start()
    
    output_file = "konusma_metni.txt"
    speech_thread = threading.Thread(target=speech_to_text_thread, 
                                   args=(output_file, 16),
                                   daemon=True)
    speech_thread.start()
    
    app.run(debug=True, threaded=True)