#HACKATHON SIC 2022-23
#BY NOVATEC.PY
#version: pyhton 3.9.12

def instalar_librerias():
    import pip
    
    pip.main(["install"], ["keras==2.11.0"])
    pip.main(["install"], ["nltk==3.7"])
    pip.main(["install"], ["numpy==1.21.5"])
    pip.main(["install"], ["keras==2.11.0"])
    pip.main(["install"], ["tensorboard==2.11.2"])
    pip.main(["install"], ["tensorflow==2.11.2"])
    pip.main(["install"], ["tensorflow-estimator==2.11.0"])
    pip.main(["install"], ["tensorflow-io-gcs-filesystem==0.29.0"])
    pip.main(["install"], ["selenium==4.7.2"])
    
    
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.dowload('universal_tagset')
    nltk.dowload('spanish_grammars')
    nltk.dowload('tagsets')
    nltk.dowload('stopwords')
    nltk.dowload('own-1.4')
    
    
if __name__ == '__main__':
    instalar_librerias()