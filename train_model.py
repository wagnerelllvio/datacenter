import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregando os dados (exemplo fictício)
# Ajustar para o seu arquivo real
dados = pd.read_csv('data/sensores.csv')  # Arquivo com as leituras dos sensores

# Separando entradas (X) e saída (y)
X = dados.drop(columns=['temperatura_objetivo'])  # tudo menos a coluna alvo
y = dados['temperatura_objetivo']                 # coluna alvo

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo
modelo = Sequential()
modelo.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(1))  # Saída

# Compilar
modelo.compile(optimizer='adam', loss='mse')

# Treinar
modelo.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1)

# Salvar o modelo
modelo.save('models/modelo_previsor.h5')

print("✅ Modelo treinado e salvo com sucesso!")

