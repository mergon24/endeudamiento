# Código de Entrenamiento - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
import xgboost as xgb
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(['Default'],axis=1)
    y_train = df[['Default']]
    print(filename, 'cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    model5 = RandomForestClassifier(n_estimators=20, max_depth=8)
    model5.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(model5, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('creditos_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()