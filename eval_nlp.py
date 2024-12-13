import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from NLP import * 

# Допустим, output — это результат вашей модели.
output=main()
# Извлекаем pooler_output и last_hidden_state
pooler_output = output.pooler_output
last_hidden_state = output.last_hidden_state

# Преобразуем в NumPy
pooler_output_np = pooler_output.detach().numpy()
last_hidden_state_np = last_hidden_state.detach().numpy()

# Визуализация pooler_output
plt.figure(figsize=(12, 6))
plt.title('Распределение значений pooler_output')
sns.histplot(pooler_output_np.flatten(), bins=30, kde=True)
plt.xlabel('Значения pooler_output')
plt.ylabel('Частота')
plt.grid()
plt.show()
