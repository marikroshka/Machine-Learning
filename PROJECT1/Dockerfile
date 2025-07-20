FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только нужные файлы
COPY app.py ./app.py
COPY model5_features.joblib ./model5_features.joblib
COPY requirements.txt ./requirements.txt
COPY .streamlit ./streamlit

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для Streamlit
EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
