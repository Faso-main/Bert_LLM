import spacy
import re
import os
import torch
from transformers import BertTokenizer, BertModel
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

class EDSSCalculator:
    def __init__(self):
        # Initializing the scores dictionary for different health metrics
        self.scores = {key: 0 for key in [
            'mobility', 'sensory', 'visual', 'bladder_bowel', 'cognitive', 
            'motor', 'cerebellar', 'speech', 'mental_state', 'fatigue'
        ]}

    def assess(self, category, level):
        scoring = {
            'sensory': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'visual': self.visual_assessment(level),
            'bladder_bowel': {'normal': 1, 'mild_incontinence': 2, 'severe_incontinence': 3},
            'cognitive': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'motor': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'cerebellar': {'normal': 1, 'mild': 2, 'severe': 3},
            'speech': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'mental_state': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'fatigue': {'none': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
        }
        
        if category in self.scores:
            if category == 'visual' and isinstance(level, tuple):
                self.scores[category] = scoring[category]
            else:
                self.scores[category] = scoring[category].get(level, 4)

    def visual_assessment(self, acuity):
        if acuity is None:
            return 4  # Assigning a high score (or "severe") when acuity data is not available
        if isinstance(acuity, tuple) and len(acuity) == 2:
            if acuity[0] >= 1.0 and acuity[1] >= 1.0:
                return 1
            elif acuity[0] >= 0.7 or acuity[1] >= 0.7:
                return 2
            else:
                return 3
        return 4  # If acuity is not valid

    def calculate_edss(self):
        total_score = sum(self.scores.values())
        return min(total_score, 30)

# Load models and initialize tokenizer
nlp = spacy.load('ru_core_news_md')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class TextExtractor:
    def __init__(self):
        self.extractors = [
            self.extract_visual_acuity,
            self.extract_motor_strength,
            self.extract_sensory_feedback,
            self.extract_bladder_bowel_function,
            self.extract_cognitive_feedback,
            self.extract_fatigue,
            self.extract_speech_condition,
            self.extract_mental_state,
            self.extract_symptoms_onset,
        ]

    def extract_information(self, text):
        results = {}
        doc = nlp(text)
        for extractor in self.extractors:
            results.update(extractor(doc))
        return results

    def extract_visual_acuity(self, doc):
        for sent in doc.sents:
            if "острота зрения" in sent.text:
                match = re.search(r'OD=(\S+);\s*OS=(\S+)', sent.text)
                if match:
                    acuity_left = float(match.group(1).replace(',', '.'))
                    acuity_right = float(match.group(2).replace(',', '.'))
                    return {'visual_acuity': (acuity_left, acuity_right)}
        return {'visual_acuity': None}

    def extract_motor_strength(self, doc):
        for sent in doc.sents:
            if "слабость" in sent.text or "неустойчивость" in sent.text:
                return {'motor_strength': 'severe'}
        return {'motor_strength': None}

    def extract_sensory_feedback(self, doc):
        for sent in doc.sents:
            if "онемение" in sent.text:
                return {'sensory_feedback': 'mild'}
        return {'sensory_feedback': None}

    def extract_bladder_bowel_function(self, doc):
        for sent in doc.sents:
            if "недержание" in sent.text or "частые позывы" in sent.text:
                return {'bladder_bowel_function': 'mild_incontinence'}
        return {'bladder_bowel_function': None}

    def extract_cognitive_feedback(self, doc):
        for sent in doc.sents:
            if "головокружение" in sent.text:
                return {'cognitive_feedback': 'mild'}
        return {'cognitive_feedback': None}

    def extract_fatigue(self, doc):
        for sent in doc.sents:
            if "утомляемость" in sent.text:
                return {'fatigue': 'moderate'}
        return {'fatigue': None}

    def extract_speech_condition(self, doc):
        for sent in doc.sents:
            if "дизартрия" in sent.text:
                return {'speech_condition': 'moderate'}
        return {'speech_condition': None}

    def extract_mental_state(self, doc):
        for sent in doc.sents:
            if "депрессия" in sent.text:
                return {'mental_state': 'moderate'}
        return {'mental_state': None}

    def extract_symptoms_onset(self, doc):
        for sent in doc.sents:
            if "заболела" in sent.text or "появилось" in sent.text:
                return {'symptoms_onset': True}
        return {'symptoms_onset': False}

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Ошибка при чтении файла: {e}")
        return ""

def evaluate_model(text):
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs

def main(user_id: str) -> None:
    file_path = os.path.join('TestFiles', f"{user_id}.txt")
    clinical_text = read_text_file(file_path)

    if not clinical_text:
        return

    # Omit the output of decoding for now to focus on extraction
    results = TextExtractor().extract_information(clinical_text)

    logging.info("Извлеченные данные: %s", results)

    edss_calculator = EDSSCalculator()
    
    if results['visual_acuity'] is not None:
        edss_calculator.assess('visual', results['visual_acuity'])
    
    if results['sensory_feedback'] is not None:
        edss_calculator.assess('sensory', results['sensory_feedback'])

    if results['bladder_bowel_function'] is not None:
        edss_calculator.assess('bladder_bowel', results['bladder_bowel_function'])

    if results['cognitive_feedback'] is not None:
        edss_calculator.assess('cognitive', results['cognitive_feedback'])

    if results['motor_strength'] is not None:
        edss_calculator.assess('motor', results['motor_strength'])

    if results['speech_condition'] is not None:
        edss_calculator.assess('speech', results['speech_condition'])

    if results['mental_state'] is not None:
        edss_calculator.assess('mental_state', results['mental_state'])

    if results['fatigue'] is not None:
        edss_calculator.assess('fatigue', results['fatigue'])

    # Calculate and log the EDSS score
    edss_score = edss_calculator.calculate_edss()
    logging.info(f'Рассчитанный уровень EDSS: {edss_score:.1f}')

    return evaluate_model(clinical_text)

if __name__ == "__main__":
    main("551")

# 708(5:5) 696(6:5) 641(7:3,5) 632(3:0) 627(3:0) 551(:)