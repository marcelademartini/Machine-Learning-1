import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tabulate import tabulate  


#carregamento da base
df = pd.read_csv('https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv')

#pré processamento
#remoção da coluna id pois é irrelevante para o modelo
df = df.drop(columns=['id'])

#conversão de letra para número
label_encoder = LabelEncoder()  
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

#features escolhidas, todas menos diagnosis e id
x = df.drop(columns=['diagnosis'])
y = df['diagnosis']

#imputação com mediana de valores ausentes nas features concavity_worts e concavity points_worst
df['concavity_mean'].fillna(df['concavity_mean'].median(), inplace=True)
df['concave points_mean'].fillna(df['concave points_mean'].median(), inplace=True)

#divisão de treinamento e teste 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

print(df.to_markdown(index=False))