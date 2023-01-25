import nltk, json, pickle
import numpy as np
import random
from nltk.stem import SnowballStemmer
from tensorflow.keras.models import load_model

lematizador = SnowballStemmer('spanish')

modelo = load_model("modelo_chatbot_pugs.h5")
intenciones = json.loads(open("intenciones.json").read())
palabras = pickle.load(open("palabras.pkl","rb"))
categorias = pickle.load(open("categorias.pkl","rb"))


def limpiar_conversacion(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=sentence_words=[lematizador.stem(palabras.lower()) for palabras in sentence_words] 
    return sentence_words

def bow (sentence,palabras,show_details=True): 
    sentence_words=limpiar_conversacion(sentence)
    
    contenedor=[0]*len(palabras)
    
    for i in sentence_words:
        for j,w in enumerate(palabras):
            if w==i: 
                contenedor[j]=1
                if show_details:
                    print("encontrado: ",w)
    return (np.array(contenedor))

def predict_class(sentence,model):
    p = bow(sentence,palabras,show_details=False) 
    res = model.predict(np.array([p]))[0] 
    
    ERROR_THRESHOLD=0.25
    results= [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD] 
    results.sort(key=lambda x: x[1], reverse=True) 
    
    return_list = []    
    for r in results:   
        return_list.append({"intent": categorias[r[0]], "probability": str(r[1])})   
    print("print de return list: ", return_list) 
    return return_list



def get_response(ints,intenciones_json):
    tag= ints[0]["intent"]
    list_of_intents=intenciones_json["intenciones"] 
    
    for i  in list_of_intents: 
        if (i["etiqueta"]==tag):
            result= random.choice(i["respuestas"]) 
            break
    return result

def chatbot_response(text): 
    ints=predict_class(text,modelo) 
    res=get_response(ints,intenciones)
    return res


    
    
def bot(texto_us):
    res = chatbot_response(texto_us)
    return res


def start_chatbot(): #esto creo que no es necesarioo
    start_intents()
    start_model()
    
    
from intents_reference import start_intents
from model_builder import start_model


if __name__ == '__main__':
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = chatbot_response(sentence)
        print(resp)

