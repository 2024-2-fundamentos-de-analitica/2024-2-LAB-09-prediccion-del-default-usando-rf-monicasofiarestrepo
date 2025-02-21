# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import pandas as pd
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, compression='zip')
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df = new_df.rename(columns={'default payment next month': 'default'})
    new_df = new_df.drop(columns=['ID'])
    new_df = new_df.loc[new_df["MARRIAGE"] != 0]
    new_df = new_df.loc[new_df["EDUCATION"] != 0]
    new_df["EDUCATION"] = new_df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return new_df
def cargar_datos(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta, index_col=False, compression='zip')

def limpiar_datos(conjunto_datos: pd.DataFrame) -> pd.DataFrame:
    conjunto_limpio = conjunto_datos.copy()
    conjunto_limpio = conjunto_limpio.rename(columns={'default payment next month': 'default'})
    conjunto_limpio = conjunto_limpio.drop(columns=['ID'])
    conjunto_limpio = conjunto_limpio.loc[conjunto_limpio["MARRIAGE"] != 0]
    conjunto_limpio = conjunto_limpio.loc[conjunto_limpio["EDUCATION"] != 0]
    conjunto_limpio["EDUCATION"] = conjunto_limpio["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return conjunto_limpio

def crear_pipeline() -> Pipeline:
    caracteristicas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocesador = ColumnTransformer(
        transformers=[
            ('categorico', OneHotEncoder(handle_unknown='ignore'), caracteristicas_categoricas)
        ],
        remainder='passthrough'
    )
    return Pipeline(steps=[('preprocesador', preprocesador), ('clasificador', RandomForestClassifier(random_state=42))])

def crear_estimador(pipeline: Pipeline) -> GridSearchCV:
    parametros_grid = {
        'clasificador__n_estimadores': [50, 100, 200],
        'clasificador__max_depth': [None, 5, 10, 20],
        'clasificador__min_samples_split': [2, 5, 10],
        'clasificador__min_samples_leaf': [1, 2, 4]
    }
    return GridSearchCV(
        pipeline,
        parametros_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
        refit=True
    )

def guardar_modelo(ruta: str, estimador: GridSearchCV):
    with gzip.open(ruta, 'wb') as archivo:
        pickle.dump(estimador, archivo)

def calcular_metricas_precision(nombre_datos: str, y, y_pred) -> dict: 
    return {
        'tipo': 'metricas',
        'dataset': nombre_datos,
        'precision': precision_score(y, y_pred, zero_division=0),
        'precision_balanceada': balanced_accuracy_score(y, y_pred),
        'recall': recall_score(y, y_pred, zero_division=0),
        'puntaje_f1': f1_score(y, y_pred, zero_division=0)
    }

def calcular_metricas_confusion(nombre_datos: str, y, y_pred) -> dict:
    matriz_confusion = confusion_matrix(y, y_pred)
    return {
        'tipo': 'matriz_confusion',
        'dataset': nombre_datos,
        'verdadero_0': {"predicho_0": int(matriz_confusion[0][0]), "predicho_1": int(matriz_confusion[0][1])},
        'verdadero_1': {"predicho_0": int(matriz_confusion[1][0]), "predicho_1": int(matriz_confusion[1][1])}
    }

def main():
    ruta_entrada = "./files/input/"
    ruta_modelos = "./files/models/"
    datos_prueba = cargar_datos(os.path.join(ruta_entrada, 'test_data.csv.zip'))
    datos_entrenamiento = cargar_datos(os.path.join(ruta_entrada, 'train_data.csv.zip'))
    datos_prueba = limpiar_datos(datos_prueba)
    datos_entrenamiento = limpiar_datos(datos_entrenamiento)
    x_prueba = datos_prueba.drop(columns=['default'])
    y_prueba = datos_prueba['default']
    x_entrenamiento = datos_entrenamiento.drop(columns=['default'])
    y_entrenamiento = datos_entrenamiento['default']
    pipeline = crear_pipeline()
    estimador = crear_estimador(pipeline)
    estimador.fit(x_entrenamiento, y_entrenamiento)
    guardar_modelo(
        os.path.join(ruta_modelos, 'model.pkl.gz'),
        estimador,
    )
    y_prueba_pred = estimador.predict(x_prueba)
    metricas_precision_prueba = calcular_metricas_precision(
        'test',
        y_prueba,
        y_prueba_pred
    )
    y_entrenamiento_pred = estimador.predict(x_entrenamiento)
    metricas_precision_entrenamiento = calcular_metricas_precision(
        'train',
        y_entrenamiento,
        y_entrenamiento_pred
    )
    metricas_confusion_prueba = calcular_metricas_confusion('test', y_prueba, y_prueba_pred)
    metricas_confusion_entrenamiento = calcular_metricas_confusion('train', y_entrenamiento, y_entrenamiento_pred)
    with open('files/output/metrics.json', 'w') as archivo:
        archivo.write(json.dumps(metricas_precision_entrenamiento)+'\n')
        archivo.write(json.dumps(metricas_precision_prueba)+'\n')
        archivo.write(json.dumps(metricas_confusion_entrenamiento)+'\n')
        archivo.write(json.dumps(metricas_confusion_prueba)+'\n')

if __name__ == "__main__":
    main()
