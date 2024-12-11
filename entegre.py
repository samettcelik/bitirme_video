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
from flask import Flask, render_template, Response, jsonify,request
import queue
import logging
import os
import speech_recognition as sr
# Eklenecek yeni importlar
from flask import send_from_directory  # Yeni eklenen
from flask_cors import CORS  # Yeni eklenen
from ses_metin import listen_and_write_segment
from question_evaluator import QuestionEvaluator



evaluator = QuestionEvaluator(api_key="APIKEY")

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
speech_active = False  # Added this line
# Global değişkenler
audio_thread = None
# Global değişkenler
stress_score = 0  # Stres skoru için global değişken
total_analyses = 0
completed_rounds = 0  # Eklendi

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
    try:
        # Ses verisini normalize et ve sınırla
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data - np.mean(audio_data)) / (np.std(audio_data) + 1e-10)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file_path = temp_file.name
            # NumPy array'i torch tensor'a çevir
            audio_tensor = torch.tensor(audio_data.T, dtype=torch.float32)
            # Ses dosyasını kaydet
            torchaudio.save(temp_file_path, audio_tensor, sampling_rate)

        # Ses verisini yükle ve işle
        speech, _ = torchaudio.load(temp_file_path)
        speech = speech.squeeze().numpy()
        
        # NaN değerleri kontrol et ve temizle
        speech = np.nan_to_num(speech, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Feature extraction için ses verisini hazırla
        inputs = feature_extractor(speech, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(device) for key in inputs}

        with torch.no_grad():
            logits = model(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        outputs = [{"Label": config.id2label[i], "Score": float(scores[i])} for i in range(len(scores))]
        
        os.unlink(temp_file_path)
        return sorted(outputs, key=lambda x: x["Score"], reverse=True)
        
    except Exception as e:
        print(f"Ses analizi hatası: {str(e)}")
        return [{"Label": "Neutral", "Score": 1.0}]  # Hata durumunda varsayılan değer
def audio_analysis():
    """Ses analizi her 5 saniyede bir gerçekleştirilir."""
    global audio_emotions, stop_analysis
    while not stop_analysis:
        if analysis_active:
            try:
                # Ses kaydı için ayrı bir stream oluştur
                with sd.InputStream(samplerate=sampling_rate, channels=1, blocksize=int(5 * sampling_rate)) as stream:
                    audio_data, _ = stream.read(int(5 * sampling_rate))
                    audio_data = np.reshape(audio_data, (-1, 1))
                    
                    if not np.isnan(audio_data).any():
                        audio_emotions = predict_audio(audio_data, sampling_rate)
                        log_to_queue(f"Ses duyguları: {audio_emotions[:2]}")
            except Exception as e:
                log_to_queue(f"Ses analizi hatası: {str(e)}")
                time.sleep(1)
        time.sleep(0.1)
def generate_frames():
    """Video akışını sağlayan generator fonksiyon"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
        
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame")
                break
                
            try:
                if analysis_active:
                    face_emotions, processed_frame = predict_face(frame)
                    generate_frames.last_emotions = face_emotions
                    frame = processed_frame
                else:
                    generate_frames.last_emotions = []
                    
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                continue
            
            time.sleep(0.01)  # Prevent excessive CPU usage
            
    except Exception as e:
        print(f"Video capture error: {str(e)}")
    finally:
        cap.release()

# Initialize the static variable
generate_frames.last_emotions = []
def speech_to_text_thread(output_file):
    """
    Enhanced speech-to-text thread optimized for single-session longer speech
    capture with advanced Turkish language support.
    """
    global speech_active, stop_analysis

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Optimize recognition parameters
    recognizer.energy_threshold = 300  # Increased sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2.0  # Allow longer pauses for natural speech
    recognizer.phrase_threshold = 0.5  # Shorter phrase threshold for better continuity
    recognizer.non_speaking_duration = 1.0  # Account for short silences

    # Initialize output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Konuşma Analizi Başladı:\n\n")

    try:
        with microphone as source:
            log_to_queue("Konuşma analizi başlıyor...")
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Longer adjustment for ambient noise

            while speech_active and not stop_analysis:
                try:
                    # Check stop signal before listening
                    if stop_analysis:
                        break

                    # Capture audio data with extended duration
                    audio_data = recognizer.listen(
                        source,
                        timeout=10,  # Longer timeout for initial silence
                        phrase_time_limit=40  # Extended phrase limit for longer sentences
                    )

                    # Check stop signal after audio capture
                    if stop_analysis:
                        break

                    try:
                        # Recognize speech with Turkish language settings
                        text = recognizer.recognize_google(
                            audio_data,
                            language="tr-TR",
                            show_all=False  # Get only the most confident result
                        )

                        # Process text (e.g., remove extra spaces)
                        processed_text = " ".join(text.split())

                        # Check stop signal before writing
                        if stop_analysis:
                            break

                        # Log recognized text
                        log_to_queue(f"Algılanan metin: {processed_text}")

                        # Write to file with a timestamp
                        timestamp = time.strftime("%H:%M:%S")
                        with open(output_file, "a", encoding="utf-8") as file:
                            file.write(f"[{timestamp}] {processed_text}\n\n")  # Add newline for readability

                    except sr.UnknownValueError:
                        if stop_analysis:
                            break
                        log_to_queue("Ses anlaşılamadı.")
                    except sr.RequestError as e:
                        if stop_analysis:
                            break
                        log_to_queue(f"Google Speech Recognition servisi hatası: {e}")

                except sr.WaitTimeoutError:
                    # Handle timeout gracefully
                    if stop_analysis:
                        break
                    log_to_queue("Dinleme zaman aşımına uğradı. Tekrar dinleniyor...")
                    continue

                # Check stop signal at end of loop
                if stop_analysis:
                    break

                time.sleep(0.1)  # Prevent high CPU usage during retries

    except Exception as e:
        log_to_queue(f"Speech recognition error: {e}")
    finally:
        # Cleanup
        speech_active = False
        log_to_queue("Konuşma analizi tamamlandı.")
        
        # Final write to file
        timestamp = time.strftime("%H:%M:%S")
        with open(output_file, "a", encoding="utf-8") as file:
            file.write(f"\n[{timestamp}] Konuşma analizi tamamlandı.\n")

def video_analysis(round_count):
    """Video analizini yöneten fonksiyon."""
    global match_count, stop_analysis, analysis_active, speech_active,stress_score,total_analyses,completed_rounds
    
    stress_score = 0
    match_count = 0
    total_analysis_count = 0
    running_results = {
        "stress_score": 0,
        "match_bonus": 0,
        "general_score": 0,
        "match_count": 0,
        "total_analyses": 0,
        "completed_rounds": 0
    }

    for i in range(1, round_count + 1):
        if stop_analysis:
            break
            
        log_to_queue(f"\n=== Döngü {i}/{round_count} başladı ===")
        start_time = time.time()
        last_analysis_time = 0
        analysis_count = 0
        round_matches = 0
        
        # Her döngü için 6 analiz yap
        while analysis_count < 6 and not stop_analysis:  
            current_time = time.time()
            
            if current_time - last_analysis_time >= 4:  # Her 4 saniyede bir analiz
                analysis_count += 1
                total_analysis_count += 1
                total_analyses = total_analysis_count
                face_emotions = getattr(generate_frames, 'last_emotions', [])
                log_to_queue(f"Analiz {analysis_count}/6 - Yüz duyguları: {face_emotions}")
                
                if audio_emotions:
                    top_audio_emotions = audio_emotions[:2]
                    log_to_queue(f"Ses duyguları: {[e['Label'] for e in top_audio_emotions]}")
                    
                    if face_emotions:
                        for face_emotion in face_emotions:
                            mapped_face_emotion = map_emotion(face_emotion)
                            audio_labels = [map_emotion(e["Label"]) for e in top_audio_emotions]
                            
                            if mapped_face_emotion in audio_labels:
                                match_count += 1
                                round_matches += 1
                                log_to_queue(f"✓ Duygu eşleşmesi başarılı! ({face_emotion} - {mapped_face_emotion})")
                                
                                # Stress score güncelleme
                                if face_emotion in ["Happy", "Surprise", "Neutral"]:
                                    stress_score += 1
                                elif face_emotion in ["Angry", "Disgust", "Fear", "Sad"]:
                                    stress_score -= 1
                
                last_analysis_time = current_time
            
            time.sleep(0.1)
        
        # Her döngü sonunda sonuçları güncelle
     # Her döngü sonunda sonuçları güncelle
        completed_rounds = i  # Global değişkeni güncelle
        running_results["completed_rounds"] = i
        running_results["total_analyses"] = total_analysis_count
        running_results["match_count"] = match_count
        running_results["match_bonus"] = match_count * 1
        running_results["stress_score"] = min(100, max(0, stress_score))
        running_results["general_score"] = min(100, running_results["stress_score"] + running_results["match_bonus"])
        
        log_to_queue(f"\n=== Döngü {i} tamamlandı ===")
        log_to_queue(f"Bu döngüdeki eşleşme sayısı: {round_matches}")
        log_to_queue(f"Toplam eşleşme sayısı: {match_count}")
        
        if i == round_count:  # Son döngüde analizi tamamla
            analysis_active = False
            speech_active = False
    
    return running_results

# Flask route'ları


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

@app.route('/api/start_analysis', methods=['GET','POST'])
def start_analysis():
    global analysis_active, stop_analysis, match_count, audio_emotions, speech_active, audio_thread,stress_score
    
    try:
        # Global değişkenleri sıfırla
        analysis_active = True
        stop_analysis = False
        match_count = 0
        audio_emotions = []
        speech_active = True
        stress_score = 0  # Stres skorunu sıfırla

        # Konuşma metni dosyasını sıfırla
        output_file = "konusma_metni.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("Analiz Başladı:\n\n")
        
        # Ses analiz thread'ini yeniden başlat
        if audio_thread and audio_thread.is_alive():
            stop_analysis = True
            audio_thread.join(timeout=1)
        
        stop_analysis = False
        audio_thread = threading.Thread(target=audio_analysis, daemon=True, name="audio_thread")
        audio_thread.start()
        
        # Konuşma analiz thread'ini başlat
        speech_thread = threading.Thread(
            target=speech_to_text_thread,
            args=(output_file,),
            daemon=True,
            name="speech_thread"
        )
        speech_thread.start()
        
        # Video analizini başlat ve sonuçları döndür
        results = video_analysis(round_count=14)
        
        return jsonify({
            "status": "success",
            "message": "Analiz tamamlandı",
            "results": results,
            "completed": True
        })
        
    except Exception as e:
        log_to_queue(f"Start analysis error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })
@app.route('/api/stop_analysis', methods=['GET', 'POST'])
def stop_analysis_route():
    global analysis_active, stop_analysis, speech_active

    try:
        # Signal stop
        stop_analysis = True
        
        # Wait for speech recognition to complete (max 5 seconds)
        wait_start = time.time()
        while speech_active and time.time() - wait_start < 5:
            time.sleep(0.1)

        # Force stop if still active
        speech_active = False
        analysis_active = False

        # Get request data
        if request.is_json:
            data = request.get_json()
        else:
            data = {}

        question_text = data.get('questionText', '')
        question_id = data.get('questionId', '')
        user_id = data.get('userId', '')

        # Small delay to ensure file writing is complete
        time.sleep(1)

        if not os.path.exists("konusma_metni.txt"):
            return jsonify({
                "status": "success",
                "message": "Analiz durduruldu fakat konuşma metni bulunamadı",
                "evaluation": None
            })

        # Evaluate speech if we have question text
        if question_text:
            evaluation_results = evaluator.evaluate_speech(question_text)
            if "error" not in evaluation_results:
                saved_results = evaluator.save_results(
                    evaluation_results,
                    question_id,
                    user_id
                )
                return jsonify({
                    "status": "success",
                    "message": "Analiz başarıyla tamamlandı ve değerlendirildi",
                    "evaluation": saved_results
                })

        return jsonify({
            "status": "success",
            "message": "Analiz durduruldu",
            "evaluation": None
        })

    except Exception as e:
        print(f"Stop analysis error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Analiz durdurulurken hata oluştu: {str(e)}"
        }), 500

    finally:
        # Ensure analysis is always stopped
        analysis_active = False
        stop_analysis = True
        speech_active = False
        
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

@app.route('/api/analysis_status', methods=['GET'])
def get_analysis_status():
    global match_count, audio_emotions, analysis_active, speech_active, stress_score  # stress_score'u ekleyelim
    
    face_emotions = getattr(generate_frames, 'last_emotions', [])
    
    current_audio_emotions = []
    if audio_emotions:
        current_audio_emotions = [
            {"emotion": e["Label"], "score": float(e["Score"])} 
            for e in audio_emotions[:2]
        ]
    
    # Analiz sonuçlarını da ekleyelim
    current_results = {
        "stress_score": stress_score,  # Stres skorunu ekledik
        "match_bonus": match_count * 1,
        "general_score": min(100, stress_score + (match_count * 1)),
        "total_analyses": total_analyses,
        "match_count": match_count
    }
    
    return jsonify({
        "active": analysis_active,
        "speech_active": speech_active,
        "match_count": match_count,
        "face_emotions": face_emotions,
        "audio_emotions": current_audio_emotions,
        "analysis_info": {
            "completed_rounds": completed_rounds,
            "total_analyses": total_analyses,
            "match_count": match_count,
            "is_complete": not analysis_active and not speech_active,
            "results": current_results  # Sonuçları ekledik
        }
    }) 
if __name__ == "__main__":
    # Ana thread'leri başlat
    audio_thread = threading.Thread(target=audio_analysis, daemon=True, name="audio_thread")
    audio_thread.start()
    
    output_file = "konusma_metni.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Analiz Sonuçları:\n\n")
    
    app.run(debug=True, threaded=True)