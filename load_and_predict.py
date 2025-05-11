import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Кількість зразків для передбачення і візуалізації
NUM_SAMPLES = 50  # ← змінюй це число для 30, 50, 100 і т.д.

# Завантаження моделі
print("[INFO] Завантаження моделі...")
model = load_model('digit_recognizer_model.h5')
print("[INFO] Модель успішно завантажена.")

# Завантаження MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Функція обробки
def preprocess_mnist(images):
    processed = []
    for img in images:
        img_rgb = array_to_img(np.stack([img]*3, axis=-1)).resize((75, 75))
        img_arr = img_to_array(img_rgb)
        img_arr = preprocess_input(img_arr)
        processed.append(img_arr)
    return np.array(processed)

# Вибірка перших NUM_SAMPLES зображень
x_samples = x_test[:NUM_SAMPLES]
y_true = y_test[:NUM_SAMPLES]
x_processed = preprocess_mnist(x_samples)

# Передбачення
print(f"[INFO] Отримання передбачень для {NUM_SAMPLES} зразків...")
predictions = model.predict(x_processed)
y_pred = np.argmax(predictions, axis=1)

# Візуалізація
cols = 5  # кількість зображень в рядку
rows = (NUM_SAMPLES + cols - 1) // cols  # розрахунок потрібної кількості рядків

plt.figure(figsize=(cols * 2.5, rows * 2.5))  # збільшуємо розмір кожного subplot

for i in range(NUM_SAMPLES):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(x_samples[i], cmap='gray')
    plt.title(f"Actual:{y_true[i]} Predicted:{y_pred[i]}", fontsize=14)  # трохи більший шрифт
    plt.axis('off')

plt.tight_layout()
plt.show()

