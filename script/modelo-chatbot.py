import numpy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import COnv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_modeld

x_train
y_train

modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(x_train[0])), activation='relu'))
modelo.add(Dropout(0,5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01,weight_decay=1e6, momentum=0.9, nesterov=True)
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

historial = modelo.fit(np.array(x_train), np.array(y_train), epochs=300, batch_size=5, verbose=1)
modelo.save("modelo_chatbot_pugs.h5")

#print("###############MODELO CREADO####################")
#reemplazar los nombres de la extensiones si es necesario
modelo = load_model("modelo_chatbot_pugs.h5")
intentos = json.loads(open("intentos.json").read())
palabras = pickle.load(open("palabras.pkl","rb"))
clases = pickle.load(open("clases.pkl","rb"))