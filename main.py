#Laboratorio3
#Stefano Aragoni, Luis Diego Santos, Carol Arevalo

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Leer el archivo de texto "entrenamiento.txt" en un DataFrame de pandas
df = pd.read_csv("entrenamiento.txt", sep='\t', header=None, names=['label', 'message'])

# Limpiar los datos de caracteres especiales en la columna "message"
df['message'] = df['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

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

def build_model(X_train, y_train):
    # Contar cuántas veces aparece cada palabra en cada clase
    ham_words = Counter()
    spam_words = Counter()
    for message, label in zip(X_train, y_train):
        words = message.split()
        if label == 0:
            ham_words.update(words)
        else:
            spam_words.update(words)
            
    # Calcular la probabilidad  de cada clase
    num_ham_messages = sum(y_train == 0)
    num_spam_messages = sum(y_train == 1)
    p_ham = num_ham_messages / len(y_train)
    p_spam = num_spam_messages / len(y_train)
    
    # Calcular la probabilidad condicional de cada palabra dado cada clase
    num_words = sum(ham_words.values()) + sum(spam_words.values())
    p_word_given_ham = {word: (count + 1) / (num_words + len(ham_words)) for word, count in ham_words.items()}
    p_word_given_spam = {word: (count + 1) / (num_words + len(spam_words)) for word, count in spam_words.items()}
    
    return p_ham, p_spam, p_word_given_ham, p_word_given_spam

def classify_message(model, message):
    p_ham, p_spam, p_word_given_ham, p_word_given_spam = model
    words = message.split()
    
    # Calcular la probabilidad de que el mensaje sea ham o spam
    log_p_ham = np.log(p_ham)
    log_p_spam = np.log(p_spam)
    for word in words:
        if word in p_word_given_ham:
            log_p_ham += np.log(p_word_given_ham[word])
        if word in p_word_given_spam:
            log_p_spam += np.log(p_word_given_spam[word])
    
    return 0 if log_p_ham > log_p_spam else 1

# Construir el modelo de Bayes con suavización de Laplace
model = build_model(X_train, y_train)

# Prueba del modelo en los datos de prueba
predictions = []
for message in X_test:
    prediction = classify_message(model, message)
    predictions.append(prediction)

# Evaluar la precisión del modelo
accuracy = np.mean(predictions == y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

#Inpunt del usuario
print("Ingrese el mensaje a clasificar: ")
mensaje = input()
result = classify_message(model, mensaje)

if result == 0:
    print("El mensaje es: Ham")
else:
    print("El mensaje es: Spam")


# Usando libreria
# Hacer split de data entre training y testing 

# Convertir el mensjae a numerico
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Entrenar el modelo 
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Ecualuar el modelo
X_test_vectorized = vectorizer.transform(X_test)
print("Accuracy:", clf.score(X_test_vectorized, y_test))
