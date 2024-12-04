import gspread, time, functools, numpy, os
from datetime import datetime, timedelta
from time import time

key__path=os.path.join("SamGTU","Key.json") 

def on_hold(seconds: int): time.sleep(seconds) #задержка для передачи излишних запросов серверу

def processing(func): #универсальный декоратор обработки ошибок
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try: return func(*args, **kwargs)
        except gspread.exceptions.APIError: on_hold(15)
        except Exception as e: print(f'Ошибка вида: {e}.....') #общая обработа ошибок
    return wrapper


class Gspread():
    @processing
    def __init__(self,key_path: str,table_tag: str) -> None:
        self.sa = gspread.service_account(key_path) #подключение в  json файлу библиотеки
        self.sh = self.sa.open(table_tag) #открытие таблицы с таким-то названием

    @processing
    def get_sheet(self, sh_id: int)->list: return self.sh.get_worksheet(sh_id).get_values()[1:]

    @processing
    def get_sheet_by_title(self, sh_tag: str)->list: return self.sh.worksheet(sh_tag).get_values()[1:]
        
    @processing
    def pass_data(self,user_data: list): #регистрация пользователя
        quetions_sh=self.sh.get_worksheet(7)
        last_row = len(quetions_sh.get_values()) + 1 #получение последнего значения заполненной строки +1 
        for col in range(1, len(user_data)+1): #заполнение (1, длинна списка ответов)
            quetions_sh.update_cell(last_row, col, user_data[col-1]) #определяем место ввода(последняя свободная, столбец, значение)


gs=Gspread(key__path,'База знаний для ЦА')


def sorting_gspread(hashmap):
    np_data = numpy.array(hashmap)
    result_dict = {row[0]: row[1].replace('\n', ' ') for row in np_data}
    return result_dict

questions_IAIT=gs.get_sheet(0)
sorted_sheet=sorting_gspread(questions_IAIT)

