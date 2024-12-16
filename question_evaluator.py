from openai import OpenAI
import os
import json
from flask import jsonify
from datetime import datetime
import requests
import re

class QuestionEvaluator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.transcript_file = "konusma_metni.txt"
        self.output_file = "degerlendirme_raporu.txt"
        
    def extract_total_score(self, evaluation_text):
        """Extract the total score from evaluation text."""
        try:
            # Look for patterns like "Toplam: 85" or "Toplam Puan: 85"
            matches = re.findall(r'Toplam\s*:?\s*(\d+)', evaluation_text)
            if matches:
                return int(matches[-1])  # Return the last matched number
            return None
        except Exception as e:
            print(f"Error extracting score: {e}")
            return None

    def evaluate_speech(self, question_text=None):
        """Evaluate speech with improved error handling and score extraction"""
        if not question_text:
            return {"error": "Question text is required"}
            
        try:
            # Read transcript with error handling
            with open(self.transcript_file, "r", encoding="utf-8") as file:
                transcript = file.read()
                if not transcript.strip():
                    return {"error": "Transcript is empty"}

            # Evaluation prompt
            prompt = f"""
            Aşağıdaki konuşma metni, belirtilen soruya bir cevaptır.
            
            Lütfen konuşmayı aşağıdaki kriterlere göre 100 üzerinden değerlendir:
            - Genel ve Doğru Bilgi (50 puan): Cevap soruya uygun mu, genel olarak verilen bilgi doğru mu?
            - Örneklerle Anlatım (30 puan): Konuyu bir veya daha fazla örnek vererek anlatmış mı?
            - Çalışma Deneyimleri (20 puan): Cevapta çalıştığı alanlardan ve kendisinin konuyla bağlantısından bir cümle dahi olsa bahsediyor mu?
            
            Not: Bu bir konuşmadan çevrilmiş metin olduğu için eksik devrik yanlış kelime ve cümlelere tolerans gösterilmelidir.
            Toplam Puanı Yazmayı Unutma ! En sona Toplam : Aldığı Puan

            Soru: {question_text}
            
            Konuşma Metni: {transcript}
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen bir uzman değerlendiricisin."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )

            result = response.choices[0].message.content
            
            # Extract and display total score
            total_score = self.extract_total_score(result)
            if total_score is not None:
                print(f"\nToplam Puan: {total_score}")

            # Save the evaluation to file
            with open(self.output_file, "w", encoding="utf-8") as file:
                file.write(result)

            return {
                "evaluation": result,
                "total_score": total_score
            }
            
        except Exception as e:
            return {"error": f"Evaluation error: {str(e)}"}

    def save_results(self, evaluation_results, question_id, user_id):
        """Save evaluation results with metadata"""
        try:
            saved_data = {
                "timestamp": datetime.now().isoformat(),
                "question_id": question_id,
                "user_id": user_id,
                "evaluation": evaluation_results,
                "source_file": self.transcript_file
            }
            
            # Save to a results file for record keeping
            results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, "w", encoding="utf-8") as file:
                json.dump(saved_data, file, ensure_ascii=False, indent=2)
                
            return saved_data
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return {
                "error": f"Failed to save results: {str(e)}",
                "partial_data": evaluation_results
            }