import pandas as pd
from sklearn import preprocessing
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score
import numpy as np



def nova_instancia(dados, colunas):
    nova_instancia = dados.loc[100, :].values.tolist() # Linha aleatória do dataset
    colunas_ = colunas[0:-1]
    nova_instancia_df = pd.DataFrame(columns = colunas)
    nova_instancia_df.loc[0] = nova_instancia
    dados_numericos_nova_instancia = nova_instancia_df.drop(columns = ['Class'])
    dados_numericos_nova_instancia = dados_numericos_nova_instancia.replace(',', '.', regex=True).apply(pd.to_numeric, errors='ignore')
    dados_classes_nova_instancia = dados['Class']
    # exit()

    pickle_in = open('dados/modelo_normalizador.pkl', 'rb')
    modelo_normalizador_treinado = pickle.load(pickle_in)
    dados_numericos_normalizados_nova_instancia = modelo_normalizador_treinado.transform(dados_numericos_nova_instancia)
    dados_numericos_normalizados_nova_instancia = pd.DataFrame(data = dados_numericos_normalizados_nova_instancia, columns = colunas_)
    print(dados_numericos_normalizados_nova_instancia)
    print(dados_classes_nova_instancia)
    modelo_treinado = open('dados/modelo_treinado.pkl', 'rb')
    print(modelo_treinado.predict(dados_numericos_normalizados_nova_instancia))
    print(modelo_treinado.predict_proba(dados_numericos_normalizados_nova_instancia))