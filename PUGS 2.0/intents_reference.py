import json

def guardar_json(datos):
    archivo = open("intenciones.json", "w")
    json.dump(datos, archivo, indent = 4)
    
    
def start_intents():
    
    biblioteca = {"intenciones":
              [
                  {"etiqueta":"saludos",
                   "patrones":["hola",
                               "hey",
                               "buenas",
                               "buenos dias",
                               "buenas noches",
                               "buenas tardes",
                               "hay alguien ahi?",
                               "que tal?"
                               "saludos",
                               "como estas?"
                              ],
                   "respuestas":["GUAU GUAU!, me llamo PUGS! su sistema de ayuda para atencion en sus preguntas o dudas sobre el centro de    atencion para su(s) mascota(s). En que te puedo ayudar?"],
                   "contexto":[""]
                },
                  
                {"etiqueta":"despedidas",
                 "patrones":["adios",
                             "hasta luego",
                             "hasta pronto",
                             "hasta la proxima",
                             "chao",
                             "bye",
                             "nos vemos"
                             ],
                 "respuestas":["Gracias por consultarme humano!! Hasta Pronto :)",
                               "Hasta la proxima! Nos vemos!",
                               "Adios, espero haberte ayudado humano :)"],
                 "contexto":[""]
                },
                  
                  
                {"etiqueta":"agradecimientos",
                   "patrones":["gracias",
                               "gracias por tu ayuda",
                               "muchas gracias",
                               "gracias por tu tiempo",
                               "te lo agradezco",
                               "has sido de gran ayuda",
                               "mil gracias",
                               "genial",
                               "excelente",
                               "brutal",
                               "muy amable"
                              ],
                   "respuestas":["Espero haberte ayudado, eres un humano genial!!",
                                "De nada, estoy para ayudarte!",
                                "Ha sido todo un placer humano!",
                                "Gracias a usted por elegirnos!"],
                   "contexto":[""]         
                  },
                  
                  
                  {"etiqueta":"servicios",
                   "patrones":["servicios",
                               "que servicios ofrecen?",
                               "de que servicios disponen?",
                               "realizan laboratorios?",
                               "realizan endoscopia?",
                               "medicina preventiva",
                               "medicina general",
                               "medicina especializada",
                               "ofrecen cirugias especializadas?",
                               "ofrecen cirugia general?",
                               "tienen servicios de acupuntura?",
                               "servicos de cardiograma",
                               "hacen ultrasonidos?",
                               "realizan electrocardiogramas?",
                               "hacen rayos x?",
                               "ofrecen servicios de urgencia?",
                               "que hacen?"
                              ],
                   "respuestas":["Ofrecemos los siguientes servicios:\n-Medicina Preventiva\n-Medicina General\n -Medicina Especializada\n -Cirugia General\n -Cirugias Especializadas\n -Acupuntura\n -Cardiogramas\n -Ultrasonidos\n -Electrocardiograma\n -Rayos X\n -Endoscopia\n -Laboratorios\n -Urgencia"],
                   "contexto":[""]         
                  },
                  
                  {"etiqueta":"horarios",
                   "patrones":["horario",
                               "que horario tienen?",
                               "a que hora atienden?",
                               "cual es su horario de atencion?",
                               "estan disponibles?",
                               "estan abiertos?",
                               "estan cerrados?",
                               "que dia atienden?",
                               "atienden?"
                              ],
                   "respuestas":["Estamos abiertos de Lunes a Sabados de 6:00 a.m - 10:00 p.m y los Domingos de 6:00 a.m - 7:00 p.m"],
                   "contexto":[""]
                   },
                  
                  {"etiqueta":"ubicacion",
                   "patrones":["ubicacion",
                               "donde se encuentran ubicados?",
                               "en donde estan localizados?",
                               "Donde estan?",
                               "Donde los puedo encontrar?",
                               "Como puedo llegar?",
                               "podrian enviar la ubicacion?",
                               "localizacion",
                               "en que parte se encuentran ubicados?"
                              ],
                   "respuestas":["Nos encontramos ubicados en Chanis, Campo Limbergh, Casa H-49, Ciudad de Panama, Panama\n En el siguiente link se muestra la ubicacion exacta: https://goo.gl/maps/tacZVtLWV5ENFv6W7"],
                   "contexto":[""]         
                  },
                  
                  {"etiqueta":"contacto",
                   "patrones":["contacto",
                               "como los puedo contactar?",
                               "tiene algun numero de contacto?",
                               "tienen whatsapp?",
                               "tienen instagram?",
                               "tienen algun numero de telefono?"
                              ],
                   "respuestas":["Puedes contactarte directamente con nosotros al Whatsapp: +507 62840188 o a nuestro numero telefónico: 391-8357 / 391-8358\n Tambien nos puedes seguir en nuestro Instagram para no perderte de ninguna promocion!!: https://www.instagram.com/clinicavetpatasyhuellas/?hl=es"],
                   "contexto":[""]         
                  },
                  
                  {"etiqueta":"sinrespuesta",
                   "patrones":[""],
                   "respuestas":["no se ha detectado respuesta",
                                 "ouh, no comprendo lo que me quieres decir humano :(",
                                 "lo siento, podrias decirlo de otro manera?, a veces me cuesta entender a los humanos"],
                   "contexto":[""]
                  }
              ]
    
            }

    guardar_json(biblioteca)
    
    
    
    
    
if __name__ == '__main__':
    start_intents()