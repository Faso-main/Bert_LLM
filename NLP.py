import spacy, re, spacy, os
from transformers import BertTokenizer, BertModel
import torch

class EDSSCalculator:
    def __init__(self):
        self.scores = {
            'mobility': 0,
            'sensory': 0,
            'visual': 0,
            'bladder_bowel': 0,
            'cognitive': 0,
            'motor': 0,
            'cerebellar': 0,
            'speech': 0,
            'mental_state': 0,
            'fatigue': 0
        }

    def assess_sensory(self, sensory_type):
        self.scores['sensory'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(sensory_type, 4)

    def assess_visual(self, acuity_left, acuity_right):
        if acuity_left >= 1.0 and acuity_right >= 1.0:
            self.scores['visual'] = 1
        elif acuity_left >= 0.7 or acuity_right >= 0.7:
            self.scores['visual'] = 2
        else:
            self.scores['visual'] = 3

    def assess_bladder_bowel(self, function):
        if function == 'normal':
            self.scores['bladder_bowel'] = 1
        elif function == 'mild_incontinence':
            self.scores['bladder_bowel'] = 2
        else:
            self.scores['bladder_bowel'] = 3

    def assess_cognitive(self, level):
        self.scores['cognitive'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(level, 4)

    def assess_motor(self, strength_level):
        self.scores['motor'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(strength_level, 4)

    def assess_cerebellar(self, coordination_level):
        self.scores['cerebellar'] = {
            'normal': 1,
            'mild': 2,
            'severe': 3,
        }.get(coordination_level, 3)

    def assess_speech(self, speech_condition):
        self.scores['speech'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(speech_condition, 4)

    def assess_mental_state(self, mental_condition):
        self.scores['mental_state'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(mental_condition, 4)

    def assess_fatigue(self, fatigue_level):
        self.scores['fatigue'] = {
            'none': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(fatigue_level, 4)

    def calculate_edss(self):
        total_score = sum(self.scores.values())
        total_score = min(total_score, 30)  # Максимум 30, соответствие докладу EDSS
        return total_score

nlp = spacy.load('ru_core_news_md')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def extract_information(text):
    results = {
        'visual_acuity': None,
        'motor_strength': None,
        'sensory_feedback': None,
        'bladder_bowel_function': None,
        'cognitive_feedback': None,
        'fatigue': None,
        'speech_condition': None,
        'mental_state': None
    }

    doc = nlp(text)

    # Извлечение информации
    for sent in doc.sents:
        if "слабость" in sent.text or "неустойчивость" in sent.text:
            results['motor_strength'] = 'severe'

        if "мурашки" in sent.text:
            results['sensory_feedback'] = 'mild'

        if "головокружение" in sent.text:
            results['cognitive_feedback'] = 'mild'

        
        if "острота зрения" in sent.text:
            match = re.search(r'OD=(\S+);\s*OS=(\S+)', sent.text)
            if match:
                results['visual_acuity'] = (float(match.group(1).replace(',', '.')), float(match.group(2).replace(',', '.')))

        if "недержание" in sent.text or "задержка" in sent.text:
            results['bladder_bowel_function'] = 'mild_incontinence'
        
        if "депрессия" in sent.text:
            results['mental_state'] = 'moderate'

        if "утомляемость" in sent.text:
            results['fatigue'] = 'moderate'

        if "дизартрия" in sent.text:
            results['speech_condition'] = 'moderate'

    return results

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return ""

def evaluate_model(text):
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # логика оценки модели
    return outputs

def main():
    file_path = os.path.join('TestFiles', '708.txt')
    clinical_text = read_text_file(file_path)

    if not clinical_text:
        return

    # Обрезка текста до 512 токенов
    tokens = tokenizer.encode(clinical_text, add_special_tokens=True, truncation=True, max_length=512)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)  # Преобразование обратно в текст

    results = extract_information(truncated_text)

    print("Извлеченные данные:")
    for key, value in results.items():
        print(f"{key}: {value}")

    edss_calculator = EDSSCalculator()

    # Заполнение данных в калькулятор
    if results['visual_acuity']:
        acuity_left, acuity_right = results['visual_acuity']
        edss_calculator.assess_visual(acuity_left, acuity_right)

    if results['sensory_feedback']:
        edss_calculator.assess_sensory(results['sensory_feedback'])

    if results['bladder_bowel_function']:
        edss_calculator.assess_bladder_bowel(results['bladder_bowel_function'])

    if results['cognitive_feedback']:
        edss_calculator.assess_cognitive(results['cognitive_feedback'])

    if results['motor_strength']:
        edss_calculator.assess_motor(results['motor_strength'])

    if results['speech_condition']:
        edss_calculator.assess_speech(results['speech_condition'])

    if results['mental_state']:
        edss_calculator.assess_mental_state(results['mental_state'])

    if results['fatigue']:
        edss_calculator.assess_fatigue(results['fatigue'])

    # Расчет итогового EDSS
    edss_score = edss_calculator.calculate_edss()

    print(f'Рассчитанный уровень EDSS: {edss_score:.1f}')

    # оценка модели
    print(f'Выход модели: {evaluate_model(truncated_text)}')

if __name__ == "__main__":
    main()
