from collections import Counter
import pandas as pd
import classificador
import nova_instancia
dados = pd.read_csv('dados\Dry_Bean_Dataset.csv', sep=';') # Dados sem colunas
colunas = ['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','ConvexArea','EquivDiameter','Extent','Solidity','roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4','Class']

dados_numericos_normalizados, dados_classes, colunas = classificador.normalizacao(dados, colunas)

# Analizar as classes depois de balanceadas
# classes_count = Counter(dados_classes)
# classes_count_n = Counter(dados_classes)
# print("Classes antes do balanceamento: ", classes_count_n)
# print("Classes depois do balanceamento: ", classes_count)

floresta = classificador.random_forest(dados_numericos_normalizados, dados_classes)
classificador.cross_validation(floresta)
# nova_instancia.nova_instancia(dados=dados, colunas=colunas)