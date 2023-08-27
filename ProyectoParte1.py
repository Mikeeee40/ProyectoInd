import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Se define una función sigmoide para poder determinar si será 1 o 0 
def sigmoid_function(X):
  return 1/(1+math.e**(-X))


#Función de la regresión
def log_regression4(X, y, alpha, epochs):
  y_ = np.reshape(y, (len(y), 1)) # shape (150,1)
  N = len(X)
  theta = np.random.randn(len(X[0]) + 1, 1) #* initialize theta
  X_vect = np.c_[np.ones((len(X), 1)), X] #* Add x0 (column of 1s)
  avg_loss_list = []
  loss_last_epoch = 9999999
  for epoch in range(epochs):
    sigmoid_x_theta = sigmoid_function(X_vect.dot(theta)) # shape: (150,5).(5,1) = (150,1)
    grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_) # shapes: (5,150).(150,1) = (5, 1)
    best_params = theta
    theta = theta - (alpha * grad)
    hyp = sigmoid_function(X_vect.dot(theta)) # shape (150,5).(5,1) = (150,1)
    avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp)
    avg_loss_list.append(avg_loss)
    loss_step = abs(loss_last_epoch - avg_loss) #*
    loss_last_epoch = avg_loss #*
  return best_params




#Se carga el dataframe con las variables a considerar
columns = ['Venue','Result','Minute', 'At_score','Type']
Messias = pd.read_csv('dataMessi.csv', usecols = columns)
df = pd.DataFrame(Messias)

#Se hace una limpieza de datos para poder asignar variables categóricas 
Donde = {"H":1, "A":2}
tipogoles = {"Left-footed shot" : 1, "Right-footed shot":2, "Header": 3, "Solo run":4, "Penalty":5, "Deflected shot on goal": 6, "Direct free kick": 7, 
             "Penalty rebound": 8, "Counter attack goal":9, "Chest": 10, "Tap-in": 11, "Long distance kick": 12, "": 13}

Messias['Venue'] = Messias['Venue'].map(Donde)
Messias['Type'] = Messias['Type'].map(tipogoles)

Messias[['valor1', 'valor2']] = Messias['Result'].str.split(':', expand=True)
Messias['valor2'] = Messias['valor2'].str.replace('[^\d.]', '', regex=True).astype(float)


#esta es la variable que se quiere predecir, por lo que hay 3 escenarios, dependiendo de si
#anoto gol o no, que gane, pierda o empate.
for i in range(len(Messias)):
    if (int(Messias["valor1"][i]) > int(Messias["valor2"][i])):
        Messias["Result"][i] = 1
    elif (int(Messias["valor1"][i]) < int(Messias["valor2"][i])):
        Messias["Result"][i] = 3
    else:
        Messias["Result"][i] = 2
        

Messias.drop(columns="valor1", inplace=True)
Messias.drop(columns="valor2", inplace=True)

#Se termina de afinar el dataser
Messias['At_score'] = Messias['At_score'].str[0]
Messias['Result'] = Messias['Result'].astype(int)
Messias['At_score'] = Messias['At_score'].astype(int)

# Define binary target 'y' dependiendo de si gana, empata o pierde
y_win = (Messias["Result"] == 1).astype(int) 
y_draw = (Messias["Result"] == 2).astype(int)
y_lose = (Messias["Result"] == 3).astype(int)
# Lista de ys
y__types = [y_win, y_draw, y_lose]
y__types = {'Win':y_win,
                'Draw':y_draw,
                'Lose':y_lose}
predicted_probs = {'Win':0.0,
                   'Draw':0.0,
                   'Lose':0.0}
actual_y = {'Win':0,
            'Draw':0,
            'Lose':0}
#Se llenan de 0 los valores que no aparecen, según yo no hay 
Messias = Messias.fillna(0)


for key, y__types in y__types.items():
  # Se separa el dataset en train y test
  X_train, X_test, y_train, y_test = train_test_split(Messias, y__types, test_size=0.2, random_state=0)
  # Escala de X
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  # Para el modelo de entrenamiento se usarán estos valores
  epochs = 1000
  alpha = 1
  best_params = log_regression4(X_train, y_train, alpha, epochs)
  # Se hacen predicciones en el dataset de entrenamiento
  index_ = 10
  X_to_predict = [list(X_test[index_])]
 
  X_to_predict = np.c_[np.ones((len(X_to_predict), 1)), X_to_predict] 
  # Se impime los resultados
  pred_probability = sigmoid_function(X_to_predict.dot(best_params))
  predicted_probs[key] = pred_probability[0][0]
  print('Our model calculated probability of sample being {}, is: {}%'.format(key, round(pred_probability[0][0]*100,2)))
  actual_y[key] = y_test[index_]

max_key = max(predicted_probs, key=predicted_probs.get)
print('\n', predicted_probs)
print('\nModel Prediction: {}'.format(max_key))
max_actual_y = max(actual_y, key=actual_y.get)
print('Real value is: {}'.format(max_actual_y))


# La programación ha sido algo bastante relevante a lo largo de los años, desde su fundación ha sido capaz de ayudar al ser
# humano a poder realizar cosas inimaginables antes, como cálculos y simulaciones. La IA ha tenido un impacto enorme dentro de las
# herramientas computacionales por lo que poder manejar la IA es una herramienta increíble dentro de las tareas diarias y no tan diarias
# A pesar de ello, es importante considerar que son modelos computacionales que se pueden complicar demasiado y que traen mucha ciencia
# e investigación por detrás.

# Machine Learning que está dentro de la IA, sirve para poder crear modelos matematicos que puedan predecir cosas a partir
# de inputs, es tan increíble que puede funcionar con múltiples variables para poder conocer la probabilidad de cierto evento. En este
# caso se quiso hacer un modelo para ver cuál era la probabilidad de que Messi perdiera, empatara o ganara un partido a partir de 
# si metio gol, en qué minuto lo metió, si estaba de local o de visita, que numero de gol fue y cómo fue que lo metió. Al hablar del
# mejor del mundo se puede esperar que en su mayoría de los partidos los haya ganado. 

# Al final, después de un ajuste de datos, una limpieza del database así como una transformación de los datos, se puede observar que es
# muy probable que Messi haya ganado un partido. El entendimiento de la regresión logística es de suma importancia para poder conocer
# cómo es que funcionan los diferentes algoritmos que aparecen en la tecnología que se usa día con día.








