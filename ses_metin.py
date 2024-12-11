import speech_recognition as sr
import time

def listen_and_write_segment(output_file, segment_id, duration):
    """
    Belirtilen süre boyunca konuşmayı dinler, metne dönüştürür ve anında dosyaya yazar.
    """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    segment_text = []  # Bu segment için toplanan metinler

    with microphone as source:
        print(f"DÖNGÜ {segment_id} dinleme başlıyor...")
        recognizer.adjust_for_ambient_noise(source)
        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            try:
                print(f"DÖNGÜ {segment_id} dinleme devam ediyor...")
                audio_data = recognizer.listen(source, timeout=duration - (time.time() - start_time))
                try:
                    # Ses tanıma
                    text = recognizer.recognize_google(audio_data, language="tr-TR")
                except sr.UnknownValueError:
                    text = "[Anlaşılmayan ses]"
                except sr.RequestError:
                    text = "[Servis hatası]"
                segment_text.append(text)
            except sr.WaitTimeoutError:
                print(f"DÖNGÜ {segment_id}: Bekleme süresi doldu, tekrar dinleniyor...")
                continue

    # Segment tamamlandıktan sonra dosyaya yaz
    with open(output_file, "a", encoding="utf-8") as file:
        file.write(f"[DÖNGÜ {segment_id}]:\n")
        file.write(" ".join(segment_text) + "\n\n")
    print(f"DÖNGÜ {segment_id} tamamlandı ve metin dosyaya yazıldı.")

if __name__ == "__main__":
    output_file = "konusma_metni.txt"

    # Dosyayı sıfırla
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Analiz:\n\n")

    # 9 döngü boyunca her biri 20 saniye dinle ve anında dosyaya yaz
    for segment in range(1, 11):  # 1'den 9'a kadar döngü
        listen_and_write_segment(output_file, segment, 16)
        print(f"DÖNGÜ {segment} tamamlandı. Bir sonraki döngüye geçiliyor...")
