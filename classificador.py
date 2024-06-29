import pandas as pd
from sklearn import preprocessing
from pickle import dump
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score
import numpy as np

def normalizacao(dados, colunas):    
    dados_numericos = dados.drop(columns = ['Class'])
    dados_classes = dados['Class']

    # Substituir vírgulas por pontos e converter as colunas para tipo numérico
    dados_numericos = dados_numericos.replace(',', '.', regex=True).apply(pd.to_numeric, errors='ignore')
    sm = SMOTE()

    dados_numericos, dados_classes = sm.fit_resample(dados_numericos, dados_classes)

    template_dados_numericos = colunas[0:-1]
    print(template_dados_numericos)
    normalizador = preprocessing.MinMaxScaler()
    modelo_normalizador = normalizador.fit(dados_numericos)
    dump(modelo_normalizador, open('dados/modelo_normalizador.pkl','wb'))

    dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)
    dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns = template_dados_numericos)

    return dados_numericos_normalizados, dados_classes, colunas


def random_forest(dados_numericos_normalizados, dados_classes):
    print(dados_numericos_normalizados)
    dados_classes = dados_classes.sample(651)
    dados_numericos_normalizados = dados_numericos_normalizados.sample(651)
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 300, num = 3)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    #Construir um objeto para responder o indutor
    forest = RandomForestClassifier() # Construir o objeto indutor

    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    
    rf_grid = GridSearchCV(forest, random_grid, refit=True, verbose=1, cv=3)

    rf_grid.fit(dados_numericos_normalizados, dados_classes)
    floresta = RandomForestClassifier(bootstrap = rf_grid.best_params_['bootstrap'], max_depth = rf_grid.best_params_['max_depth'], max_features= rf_grid.best_params_['max_features'], min_samples_leaf= rf_grid.best_params_['min_samples_leaf'], min_samples_split= rf_grid.best_params_['min_samples_split'], n_estimators = rf_grid.best_params_['n_estimators'])
    floresta.fit(dados_numericos_normalizados, dados_classes)
    dump(floresta, open('dados/modelo_treinado.pkl', 'wb'))
    return floresta

def cross_validation(floresta, dados_numericos_normalizados, dados_classes):
    scoring = ['precision_macro', 'recall_macro']
    scores_cross = cross_validate(floresta, dados_numericos_normalizados, dados_classes.values.ravel(), scoring=scoring)
    print(scores_cross['test_precision_macro'].mean())
    print(scores_cross['test_recall_macro'].mean())

    x_ = np.array(["test_precision_macro", "test_recall_macro"])
    y_ = np.array([scores_cross['test_precision_macro'].mean(), scores_cross['test_recall_macro'].mean()])
    print(x_)
    print(y_)
