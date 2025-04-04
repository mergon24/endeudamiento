
# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)).set_index('ID')
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    # Convertimos SEX en dummy
    df.SEX = df.SEX-1
    # Generar variables Cuantitativas transformadas
    LIST_BILL = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    # Reemplazamos los valores -1 por 0
    for i in LIST_BILL:
     df.loc[df.loc[:,i] == -1,i] = 0
    # Calculamos el logaritmo para cada variable numérica continua
    df['LOG_BILL_AMT1'] = round(np.log1p(df['BILL_AMT1']),5)
    df['LOG_BILL_AMT2'] = round(np.log1p(df['BILL_AMT2']),5)
    df['LOG_BILL_AMT3'] = round(np.log1p(df['BILL_AMT3']),5)
    df['LOG_BILL_AMT4'] = round(np.log1p(df['BILL_AMT4']),5)
    df['LOG_BILL_AMT5'] = round(np.log1p(df['BILL_AMT5']),5)
    df['LOG_BILL_AMT6'] = round(np.log1p(df['BILL_AMT6']),5)
    df['LOG_PAY_AMT1'] = round(np.log1p(df['PAY_AMT1']),5)
    df['LOG_PAY_AMT2'] = round(np.log1p(df['PAY_AMT2']),5)
    df['LOG_PAY_AMT3'] = round(np.log1p(df['PAY_AMT3']),5)
    df['LOG_PAY_AMT4'] = round(np.log1p(df['PAY_AMT4']),5)
    df['LOG_PAY_AMT5'] = round(np.log1p(df['PAY_AMT5']),5)
    df['LOG_PAY_AMT6'] = round(np.log1p(df['PAY_AMT6']),5)
    # Generamos listas de las variables para utilizarlas en los cálculos agrupados
    LIST_PAY  = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    LIST_BILL = ['LOG_BILL_AMT1','LOG_BILL_AMT2','LOG_BILL_AMT3','LOG_BILL_AMT4','LOG_BILL_AMT5','LOG_BILL_AMT6']
    LIST_PAMT = ['LOG_PAY_AMT1','LOG_PAY_AMT2','LOG_PAY_AMT3','LOG_PAY_AMT4','LOG_PAY_AMT5','LOG_PAY_AMT6']
    # Reemplazar los valores faltantes con cero
    for i in LIST_BILL:
     df.loc[df.loc[:,i].isnull(),i] = 0
    # Creamos otras variables derivadas
    df['STD_PAY_TOT']    = df[LIST_PAY].std(axis=1)
    df['CANT_PAY_MAY0']  = df[LIST_PAY].gt(0).sum(axis=1)
    df['AVG_LBILL_TOT']  = df[LIST_BILL].mean(axis=1)
    df['STD_LBILL_TOT']  = df[LIST_BILL].std(axis=1)
    df['CV_LBILL_TOT']   =  df['STD_LBILL_TOT']/(df['AVG_LBILL_TOT']+1)
    df['AVG_LPAY_TOT']   = df[LIST_PAMT].mean(axis=1)
    df['STD_LPAY_TOT']   = df[LIST_PAMT].std(axis=1)
    df['CV_LPAY_TOT']    =  df['STD_LPAY_TOT']/(df['AVG_LPAY_TOT']+1)
    df['AVG_EXP_1'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +
                     (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4'])) +
                     (df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3'])) +
                     (df['BILL_AMT2'] - (df['BILL_AMT3'] - df['PAY_AMT2'])) +
                     (df['BILL_AMT1'] - (df['BILL_AMT2'] - df['PAY_AMT1']))) / 5) / df['LIMIT_BAL']
    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('defaultcc.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['SEX','PAY_1','AGE','LIMIT_BAL','CV_LPAY_TOT','CV_LBILL_TOT','CANT_PAY_MAY0','BILL_AMT1','LOG_BILL_AMT1','AVG_LPAY_TOT','STD_PAY_TOT','AVG_EXP_1','DEFAULT'],'credit_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('defaultcc_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['SEX','PAY_1','AGE','LIMIT_BAL','CV_LPAY_TOT','CV_LBILL_TOT','CANT_PAY_MAY0','BILL_AMT1','LOG_BILL_AMT1','AVG_LPAY_TOT','STD_PAY_TOT','AVG_EXP_1','DEFAULT'],'credit_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('defaultcc_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['SEX','PAY_1','AGE','LIMIT_BAL','CV_LPAY_TOT','CV_LBILL_TOT','CANT_PAY_MAY0','BILL_AMT1','LOG_BILL_AMT1','AVG_LPAY_TOT','STD_PAY_TOT','AVG_EXP_1'],'credit_score.csv')
    
if __name__ == "__main__":
    main()
