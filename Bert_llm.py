import torch, time, numpy
from transformers import BertTokenizer, BertModel
from SamGTU.Gspread_NY import sorted_sheet


class Inquiry: # принимает словарь{"вопрос":"ответ"}
    def __init__(self, hashmap: dict):
        self.hashmap = hashmap

    def get_questions(self): return list(self.hashmap.keys())
    
    def get_answer(self, question): return self.hashmap.get(question, None)

class BertEmbedding: # класс инициализации и настройки модели
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, question: str):
        inumpyuts = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad(): outputs = self.model(**inumpyuts)
        return outputs.last_hidden_state.mean(dim=1)

class Search: # ответ и настройки вывода
    def __init__(self, hashmap: Inquiry, confidence: float, track_time: bool, show_accuracy: bool):
        self.hashmap = hashmap
        self.confidence = confidence
        self.track_time = track_time
        self.show_accuracy = show_accuracy
        self.bert_encoder = BertEmbedding()

    def find_answer(self, user_question: str):
        if self.track_time: start_time = time.time()

        user_embedding = self.bert_encoder.encode(user_question)
        best_match_index = -1
        best_similarity = -1

        for index, question in enumerate(self.hashmap.get_questions()):
            question_embedding = self.bert_encoder.encode(question)
            similarity = torch.cosine_similarity(user_embedding, question_embedding).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = index

        if self.track_time:
            processing_time = time.time() - start_time 
            time_info = f'\nВремя обработки запроса: {processing_time:.2f} секунд'
        else: time_info = ''

        if best_similarity > self.confidence:
            matched_question = self.hashmap.get_questions()[best_match_index]
            response = self.hashmap.get_answer(matched_question)
            accuracy_info = f'\nКоэффициент точности: {best_similarity:.2f}' if self.show_accuracy else ''
            return f'{response}{time_info}{accuracy_info}'
        else:
            return f'Извините, не могу ответить на ваш вопрос.{time_info}'

#(отсортированый словарь, уверенность, считать время?, показать точность выбранного ответа?)
response = Search(Inquiry(sorted_sheet), 0.7, True, True) 

while True:
    try: print(response.find_answer(input("Введите ваш вопрос: ")))
    except KeyboardInterrupt:
        print(f'\nРабота приостановлена вручную..............')
        break
    except Exception as e: print(f'\nОшибка вида: {e}')
