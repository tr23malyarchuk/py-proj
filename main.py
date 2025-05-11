import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from colorama import init, Fore
init(autoreset=True)

print("[INFO] Запуск програми...")

# Завантажуємо MNIST
print("[INFO] Завантаження датасету MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"[INFO] MNIST завантажено: {x_train.shape[0]} тренувальних прикладів, {x_test.shape[0]} тестових.")

# Перетворюємо 28x28 → 75x75 RGB + нормалізація
def preprocess_mnist(images):
    processed = []
    for img in tqdm(images, desc=Fore.GREEN + "Обробка зображень"):
        img_rgb = array_to_img(np.stack([img]*3, axis=-1)).resize((75, 75))  # робимо 3 канали + зміна розміру
        img_arr = img_to_array(img_rgb)  # переводимо у масив numpy
        img_arr = preprocess_input(img_arr)  # нормалізація як для InceptionV3
        processed.append(img_arr)
    return np.array(processed)

# Обробка тренувального та тестового набору
print("[INFO] Початок обробки тренувального набору...")
x_train_resized = preprocess_mnist(x_train)
print("[INFO] Початок обробки тестового набору...")
x_test_resized = preprocess_mnist(x_test)
print("[INFO] Обробка завершена.")

# Завантажуємо InceptionV3 без верхніх (класифікаційних) шарів
print("[INFO] Завантаження базової моделі InceptionV3...")
base_model = InceptionV3(include_top=False, input_tensor=Input(shape=(75, 75, 3)), weights='imagenet')
base_model.trainable = False  # заморожуємо ваги

# Додаємо власні шари класифікації
x = base_model.output
x = GlobalAveragePooling2D()(x)  # усереднення ознак
x = Dense(128, activation='relu')(x)  # прихований шар
predictions = Dense(10, activation='softmax')(x)  # 10 класів для цифр

# Створення та компіляція моделі
print("[INFO] Створення та компіляція повної моделі...")
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')])
print("[INFO] Компіляція завершена.")

# Навчання моделі
print("[INFO] Початок навчання...")
model.fit(x_train_resized, y_train, epochs=5, batch_size=64, validation_split=0.1)
print("[INFO] Навчання завершено.")

# Підготовка одного прикладу для передбачення
print("[INFO] Передбачення випадкового прикладу...")
i = np.random.randint(len(x_test))
image = x_test[i]
true_label = y_test[i]

# Підготовка вхідних даних для моделі
img_rgb = array_to_img(np.stack([image]*3, axis=-1)).resize((75, 75))
input_tensor = preprocess_input(img_to_array(img_rgb))
input_tensor = np.expand_dims(input_tensor, axis=0)  # додавання batch-виміру

# Отримання передбачення
pred = model.predict(input_tensor)
predicted_digit = np.argmax(pred)
confidence = np.max(pred) * 100

# Візуалізація результату
plt.imshow(image, cmap='gray')
plt.title(f"True: {true_label}, Predicted: {predicted_digit} ({confidence:.2f}%)")
plt.axis('off')
plt.show()

# Збереження моделі
print("[INFO] Збереження моделі у файл digit_recognizer_model.h5...")
model.save('digit_recognizer_model.h5')
print("[INFO] Модель збережено успішно.")

# Оцінка точності моделі на повному тестовому наборі
print("[INFO] Оцінка якості моделі (classification report)...")
y_pred_probs = model.predict(x_test_resized)
y_pred = np.argmax(y_pred_probs, axis=1)

# Виведення precision, recall, f1-score
print("\n[Звіт класифікації (sklearn)]")
print(classification_report(y_test, y_pred, digits=4))

# Матриця помилок
print("[Матриця помилок]")
print(confusion_matrix(y_test, y_pred))

print("[INFO] Програма завершила роботу.")
