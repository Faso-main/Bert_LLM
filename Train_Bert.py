import torch
import time
import logging
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from SamGTU.Gspread_NY import sorted_sheet

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Inquiry:
    def __init__(self, hashmap: dict):
        self.hashmap = hashmap

    def get_questions(self): 
        return list(self.hashmap.keys())
    
    def get_answer(self, question): 
        return self.hashmap.get(question)

class BertEmbedding:
    def __init__(self, model_name='DeepPavlov/rubert-base-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.cache = {}  # Кэшируем векторы

    def encode(self, text: str):
        if text in self.cache:
            return self.cache[text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        self.cache[text] = embedding
        return embedding

class Search:
    def __init__(self, hashmap: Inquiry, confidence: float, track_time: bool, show_accuracy: bool):
        self.hashmap = hashmap
        self.confidence = confidence
        self.track_time = track_time
        self.show_accuracy = show_accuracy
        self.bert_encoder = BertEmbedding()
        self.question_embeddings = {question: self.bert_encoder.encode(question) for question in self.hashmap.get_questions()}

    def find_answer(self, user_question: str):
        if self.track_time:
            start_time = time.time()
        
        user_embedding = self.bert_encoder.encode(user_question)
        best_match_index = -1
        best_similarity = -1

        for index, (question, question_embedding) in enumerate(self.question_embeddings.items()):
            similarity = torch.cosine_similarity(user_embedding, question_embedding).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = index

        time_info = f'\nВремя обработки: {time.time() - start_time:.2f} секунд' if self.track_time else ''
        
        if best_similarity > self.confidence:
            matched_question = self.hashmap.get_questions()[best_match_index]
            response = self.hashmap.get_answer(matched_question)
            accuracy_info = f'\nТочность: {best_similarity:.2f}' if self.show_accuracy else ''
            return f'{response}{time_info}{accuracy_info}'
        else:
            return f'Извините, я не могу ответить на ваш вопрос.{time_info}'

class BertTrainer:
    def __init__(self, model_name='DeepPavlov/rubert-base-cased', num_epochs=1, learning_rate=1e-5):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.train()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CosineSimilarity(dim=-1)

    def train(self, questions, answers):
        for epoch in range(self.num_epochs):
            total_loss = 0
            for question in questions:
                question_embedding = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
                question_embedding = self.model(**question_embedding).last_hidden_state.mean(dim=1)

                answer_embedding = self.tokenizer(answers[question], return_tensors="pt", padding=True, truncation=True).to(self.device)
                answer_embedding = self.model(**answer_embedding).last_hidden_state.mean(dim=1)

                loss = 1 - self.criterion(question_embedding, answer_embedding).mean()
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {total_loss:.4f}')

# Инициализация данных
inquiry = Inquiry(sorted_sheet)
trainer = BertTrainer(num_epochs=2)
questions = inquiry.get_questions()
answers = inquiry.hashmap
trainer.train(questions, answers)

# Поиск ответов
response = Search(inquiry, confidence=0.7, track_time=True, show_accuracy=True) 

while True:
    try:
        user_question = input("Введите ваш вопрос: ")
        print(response.find_answer(user_question))
    except KeyboardInterrupt:
        print(f'\nПроцесс прерван...')
        break
    except Exception as e:
        logging.error(f'Ошибка: {e}')
