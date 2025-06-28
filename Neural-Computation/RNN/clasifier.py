import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset

# Construcción automatica del dataframe
# Se extraen los videos para entrenamiento
dataset_path = os.listdir('dataset/train')
label_types = os.listdir('dataset/train')
rooms = []
for item in dataset_path:
    # Se obtienen todos los videos de entrenamiento
    all_rooms = os.listdir('dataset/train' + '/' +item)

    # Se añaden a la lista de entrenamiento
    for room in all_rooms:
        rooms.append((item, str('dataset/train' + '/' +item) + '/' + room))
    
# Se genera la el archivo con la lista de videos de entrenamiento   
train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
df = train_df.loc[:,['video_name','tag']]
df.to_csv('train.csv')

# Se extraen los videos para test
dataset_path = os.listdir('dataset/test')
room_types = os.listdir('dataset/test')
rooms = []
for item in dataset_path:
    # Se obtienen todos los videos de test
    all_rooms = os.listdir('dataset/test' + '/' +item)

    # Se añaden a la lista de test
    for room in all_rooms:
        rooms.append((item, str('dataset/test' + '/' +item) + '/' + room))
    
# Se genera la el archivo con la lista de videos de test      
test_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
df = test_df.loc[:,['video_name','tag']]
df.to_csv('test.csv')

# Se utiliza la memoria de video de la GPU para optimizar el uso de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

# Cargar de datos
train_csv_path = './train.csv'
test_csv_path = './test.csv'
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Parámetros de preprocesamiento
desired_frame_size = (120, 120)  # Tamaño de los fotogramas
desired_fps = 10  # Fotogramas por segundo para la extracción

# Función para extraer fotogramas de un video
def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, desired_frame_size)
        frame = frame / 255.0  # Normalización
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    # Completar la secuencia de frames si es necesario
    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))

    return np.array(frames)

# Función para preprocesar los datos
def preprocess_data(df, base_path):
    X = []
    y = []

    for index, row in df.iterrows():
        video_path = row['video_name']
        label = row['tag']
        frames = extract_frames(os.path.join(base_path, video_path))
        if frames.shape[0] == 0:  # Filtra videos sin frames
            continue
        X.append(frames)
        y.append(label)

    # Convertir a arrays de numpy
    X = np.array(X)
    y = np.array(y)
    return X, y

# Codificar etiquetas y dividir en datos de entrenamiento y prueba
labels = train_df['tag'].unique()
label_to_index = {label: idx for idx, label in enumerate(labels)}
train_df['label'] = train_df['tag'].map(label_to_index)
test_df['label'] = test_df['tag'].map(label_to_index)  # Asegúrate de mapear en test_df también

# Preprocesamiento de datos de entrenamiento y prueba
X_train, y_train = preprocess_data(train_df, './')
X_test, y_test = preprocess_data(test_df, './')

# Convertir etiquetas a números enteros antes de one-hot encoding
y_train = [label_to_index[label] for label in y_train]
y_test = [label_to_index[label] for label in y_test]

# Convertir las etiquetas a formato one-hot
y_train = to_categorical(y_train, num_classes=len(labels))
y_test = to_categorical(y_test, num_classes=len(labels))

# Optimizar la carga de datos con tf.data.Dataset
train_dataset = Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(4).prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = Dataset.from_tensor_slices((X_test, y_test)).batch(4).prefetch(buffer_size=tf.data.AUTOTUNE)

# Construcción del modelo CNN + LSTM (RNN) con una arquitectura más simple
model = models.Sequential()

# Capas CNN para extraer características de los fotogramas
# Se emplean convolucionales 2D + capas de pooling para reducir dimensionalidad 
model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu'), input_shape=(30, 120, 120, 3)))
model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
# Capa final de CNN, aplana los datos para que se puedan utilizar en la siguiente red (RNN)
model.add(layers.TimeDistributed(layers.Flatten()))

# Capa LSTM (RNN) para modelar la secuencia de características
model.add(layers.LSTM(32, return_sequences=False))

# Capa densa final para clasificación
model.add(layers.Dense(len(labels), activation='softmax'))

# Se compila del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo con el uso de la GPU
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluación del modelo
loss, accuracy = model.evaluate(test_dataset)
print(f'Pérdida en test: {loss:.4f}, Precisión en test: {accuracy:.4f}')

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Convertir las predicciones de one-hot a etiquetas numéricas
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Obtener nombres de las clases (en caso de que no estén definidos)
class_names = label_to_index.keys()

# Crear un DataFrame para almacenar los resultados
results = pd.DataFrame(columns=['Clase', 'Aciertos', 'Fallos'])

# Calcular aciertos y fallos por clase
for class_name in class_names:
    class_index = label_to_index[class_name]
    
    # Contar el número de aciertos e incorrectos para la clase actual
    correct = np.sum((y_pred_classes == y_true_classes) & (y_true_classes == class_index))
    incorrect = np.sum((y_pred_classes != y_true_classes) & (y_true_classes == class_index))
    
    # Agregar los resultados al DataFrame
    new_row = pd.DataFrame([{'Clase': class_name, 'Aciertos': correct, 'Fallos': incorrect}])
    results = pd.concat([results, new_row], ignore_index=True)

# Mostrar la tabla de resultados
print(results)