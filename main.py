#Laboratorio3
#Stefano Aragoni, Luis Diego Santos, Carol Arevalo

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#=======================================Task 1.1 - Lectura y limpieza del dataset=========================================================

# Leer el archivo de texto "entrenamiento.txt" en un DataFrame de pandas
df = pd.read_csv("entrenamiento.txt", sep='\t', header=None, names=['label', 'message'])

# Limpiar los datos de caracteres especiales en la columna "message"
df['message'] = df['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

#eliminar numeros
df['message'] = df['message'].apply(lambda x: re.sub(r'\d+', '', x))

# Convertir el mensaje a minúsculas
df['message'] = df['message'].str.lower()

# Convertir la columna "label" a 1 o 0
df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Mostrar los primeros 5 registros del DataFrame
print(df.head())

# Separar el dataset en un 80% para entrenamiento y un 20% para pruebas
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

print("Cantidad de registros en el conjunto de entrenamiento: ", len(X_train))

print("Cantidad de registros en el conjunto de pruebas: ", len(X_test))

#=======================================Task 1.2 Construcción del modelo=========================================================

def build_model(X_train, y_train):
    ham_words = []
    spam_words = []

    for message, label in zip(X_train, y_train):
        words = message.split()
        if label == 0:
            for word in words:
                ham_words.append(word)
        else:
            for word in words:
                spam_words.append(word)
               
    # Count the frequency of each word in the ham and spam messages
    ham_word_counts = Counter(ham_words)
    spam_word_counts = Counter(spam_words)
    
    # Calcular la probabilidad  de cada clase
    cant_ham = len(ham_words)
    cant_spam = len(spam_words)
    total_words = cant_ham + cant_spam
    probability_ham = cant_ham / total_words
    probability_spam = cant_spam / total_words
    
    return {'ham': probability_ham, 'spam': probability_spam, 'ham_word_counts': ham_word_counts, 'spam_word_counts': spam_word_counts}

def classify_message(model, message):
    words = message.split()
    probability_ham = model['ham']
    probability_spam = model['spam']
    ham_word_counts = model['ham_word_counts']
    spam_word_counts = model['spam_word_counts']
    
    # Estimate the probability of each word being ham or spam
    for word in words:
        ham_count = ham_word_counts.get(word, 0)
        spam_count = spam_word_counts.get(word, 0)
        total_count = ham_count + spam_count
        word_prob_ham = (ham_count + 1) / (total_count + 2)
        word_prob_spam = (spam_count + 1) / (total_count + 2)
        
        probability_ham *= word_prob_ham
        probability_spam *= word_prob_spam
    
    if probability_ham > probability_spam:
        return 0
    else:
        return 1


# Construir el modelo de Bayes con suavización de Laplace
model = build_model(X_train, y_train)

######### SUBSET TRAIN #########

# Prueba del modelo en los datos de prueba
predictions = []
for message in X_train:
    prediction = classify_message(model, message)
    predictions.append(prediction)

# Evaluar la precisión del modelo
correct = 0
for prediction, label in zip(predictions, y_train):
    if prediction == label:
        correct += 1

print("\nExactitud (Modelo Propio) (Training):", correct / len(predictions))

######### SUBSET TEST #########

# Prueba del modelo en los datos de prueba
predictions = []
for message in X_test:
    prediction = classify_message(model, message)
    predictions.append(prediction)

# Evaluar la precisión del modelo
correct = 0
for prediction, label in zip(predictions, y_test):
    if prediction == label:
        correct += 1
        
print("Exactitud (Modelo Propio) (Test):", correct / len(predictions))


#=======================================Task 1.3 Clasificacion de mensajes futuros=========================================================
#Inpunt del usuario
while True:
    print("\nIngrese el mensaje a clasificar: ")
    mensaje = input()
    
    # Limpiar los datos de caracteres especiales en la columna "message"
    mensaje = re.sub(r'[^\w\s]', '', mensaje)
    mensaje = mensaje.lower()
    
    result = classify_message(model, mensaje)
    
    if result == 0:
        print("\nEl mensaje es: Ham")
    else:
        print("\nEl mensaje es: Spam")
        
    print("\nDesea continuar? (Y/N) (A continuacion se mostrara la exactitud del modelo generado con SKLEARN)")
    continuar = input()
    if continuar == "N" or continuar == "n":
        break


#=======================================Task 1.4 - Comparacion de Librerias========================================================
# Usando libreria
# Hacer split de data entre training y testing 

# Convertir el mensjae a numerico
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Entrenar el modelo 
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Ecualuar el modelo

######### SUBSET TRAIN #########
X_train_vectorized = vectorizer.transform(X_train)
print("\nExactitud (Libreria):", clf.score(X_train_vectorized, y_train))

######### SUBSET TEST #########
X_test_vectorized = vectorizer.transform(X_test)
print("Exactitud *():", clf.score(X_test_vectorized, y_test))




