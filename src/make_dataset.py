
# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os

# Leemos los archivos csv
def read_file_csv(filename):
    filepath = os.path.join('../data/raw/', filename)  # Une la ruta correctamente
    df = pd.read_csv(filepath, sep=";", na_values=['NA'], encoding="latin1")  # Lee el CSV correctamente
    print(filename, 'cargado correctamente')
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    # Eliminar espacios en los nombres de columna
    df.columns = df.columns.str.strip()
    # Verificar si existe la columna antes de usarla
    if 'Nro_dependiente' in df.columns:
        df['Nro_dependiente'] = df['Nro_dependiente'].fillna(0)
    else:
        print("⚠️  La columna 'Nro_dependiente' no existe en el DataFrame")
    
    return df
    #REEMPLAZAR LOS VALORES NULOS CON 0 DE LA COLUMNA 'Nro_dependiente'
    df_isnull2 = df['Nro_dependiente'].fillna(0)
    df['Nro_dependiente'] = df_isnull2
    
    #APLICAR OUTLIER CAPPING
    def outlier_capping(x):
        x = x.clip(upper=x.quantile(0.95))
        return(x)

    df_creditos_tratamiento = df[['Prct_uso_tc']].apply(lambda x: outlier_capping(x))
    df = pd.concat([df.drop(columns=['Prct_uso_tc']), df_creditos_tratamiento], axis=1,)
    df_creditos_tratamiento2 = df[['Prct_deuda_vs_ingresos']].apply(lambda y: outlier_capping(y))
    df = pd.concat([df.drop(columns=['Prct_deuda_vs_ingresos']), df_creditos_tratamiento2], axis=1,)
    df_creditos_tratamiento3 = df[['Mto_ingreso_mensual']].apply(lambda z: outlier_capping(z))
    df = pd.concat([df.drop(columns=['Mto_ingreso_mensual']), df_creditos_tratamiento3], axis=1,)
    df_creditos_tratamiento4 = df[['Nro_prestao_retrasados']].apply(lambda a: outlier_capping(a))
    df = pd.concat([df.drop(columns=['Nro_prestao_retrasados']), df_creditos_tratamiento4], axis=1,)
    df_creditos_tratamiento5 = df[['Nro_retraso_60dias']].apply(lambda b: outlier_capping(b))
    df = pd.concat([df.drop(columns=['Nro_retraso_60dias']), df_creditos_tratamiento5], axis=1,)
    df_creditos_tratamiento6 = df[['Nro_creditos_hipotecarios']].apply(lambda c: outlier_capping(c))
    df = pd.concat([df.drop(columns=['Nro_creditos_hipotecarios']), df_creditos_tratamiento6], axis=1,)
    df_creditos_tratamiento7 = df[['Nro_retraso_ultm3anios']].apply(lambda d: outlier_capping(d))
    df = pd.concat([df.drop(columns=['Nro_retraso_ultm3anios']), df_creditos_tratamiento7], axis=1,)
      
    #ARREGLO DE LOS VALORES EN 0 DE LA COLUMNA 'Mto_ingreso_mensual'
    def monto_fix(monto):
        if monto==0:
            return np.nan
        return monto

    df['Mto_ingreso_mensual'] = df['Mto_ingreso_mensual'].apply(monto_fix)
    
    #HALLAR EL PROMEDIO Y DESVIACIÓN ESTÁNDAR PARA DEFINIR EL INTERVALO DE CONFIANZA
    age_avg = df['Mto_ingreso_mensual'].mean()
    age_std = df['Mto_ingreso_mensual'].std()
    age_null_count = df['Mto_ingreso_mensual'].isnull().sum()
    if age_null_count > 0:  # Solo ejecutamos si hay valores nulos
        age_null_random_list = np.random.randint(
            low=int(age_avg - age_std), 
            high=int(age_avg + age_std), 
            size=age_null_count  # Aseguramos que el tamaño sea igual a los valores nulos
        )

    df.loc[df['Mto_ingreso_mensual'].isnull(), 'Mto_ingreso_mensual'] = age_null_random_list
    
    df['Mto_ingreso_mensual'] = df['Mto_ingreso_mensual'].astype(int)
    print("Monto Promedio: " + str(age_avg))
    print("Desvió Std Monto: " + str(age_std))
    print("Intervalo para asignar monto aleatoria: " + str(int(age_avg - age_std)) + " a " + str(int(age_avg + age_std)))
    
    # **5. Agrupación por rangos para aplicación de Machine Learning**
    
    # MAPPING 'Edad' DE LOS CLIENTES
    df.loc[ df['Edad'] <= 24, 'EdadEncoded'] = 0
    df.loc[(df['Edad'] > 24) & (df['Edad'] <= 40), 'EdadEncoded'] = 1
    df.loc[(df['Edad'] > 40) & (df['Edad'] <= 60), 'EdadEncoded'] = 2
    df.loc[(df['Edad'] > 60) & (df['Edad'] <= 80), 'EdadEncoded'] = 3
    df.loc[ df['Edad'] > 80, 'EdadEncoded'] = 4
    
    # MAPPING 'Nro_prestao_retrasados' DE LOS CLIENTES
    df.loc[ df['Nro_prestao_retrasados'] <= 0, 'RetrasosEncoded'] = 0
    df.loc[(df['Nro_prestao_retrasados'] > 0) & (df['Nro_prestao_retrasados'] <= 2), 'RetrasosEncoded'] = 1
    df.loc[(df['Nro_prestao_retrasados'] > 2) & (df['Nro_prestao_retrasados'] <= 5), 'RetrasosEncoded'] = 2
    df.loc[(df['Nro_prestao_retrasados'] > 5) & (df['Nro_prestao_retrasados'] <= 8), 'RetrasosEncoded'] = 3
    df.loc[(df['Nro_prestao_retrasados'] > 8) & (df['Nro_prestao_retrasados'] <= 13), 'RetrasosEncoded'] = 4
    df.loc[ df['Nro_prestao_retrasados'] > 13, 'RetrasosEncoded'] = 5
    
    # MAPPING 'Nro_prod_financieros_deuda' DE LOS CLIENTES
    df.loc[ df['Nro_prod_financieros_deuda'] <= 0, 'FinancierosEncoded'] = 0
    df.loc[(df['Nro_prod_financieros_deuda'] > 0) & (df['Nro_prod_financieros_deuda'] <= 15), 'FinancierosEncoded'] = 1
    df.loc[(df['Nro_prod_financieros_deuda'] > 15) & (df['Nro_prod_financieros_deuda'] <= 25), 'FinancierosEncoded'] = 2
    df.loc[(df['Nro_prod_financieros_deuda'] > 25) & (df['Nro_prod_financieros_deuda'] <= 35), 'FinancierosEncoded'] = 3
    df.loc[(df['Nro_prod_financieros_deuda'] > 35) & (df['Nro_prod_financieros_deuda'] <= 45), 'FinancierosEncoded'] = 4
    df.loc[ df['Nro_prod_financieros_deuda'] > 45, 'FinancierosEncoded'] = 5
    
    # MAPPING 'Nro_retraso_60dias' DE LOS CLIENTES
    df.loc[ df['Nro_retraso_60dias'] <= 0, '60Enconded'] = 0
    df.loc[(df['Nro_retraso_60dias'] > 0) & (df['Nro_retraso_60dias'] <= 3), '60Enconded'] = 1
    df.loc[(df['Nro_retraso_60dias'] > 3) & (df['Nro_retraso_60dias'] <= 7), '60Enconded'] = 2
    df.loc[(df['Nro_retraso_60dias'] > 7) & (df['Nro_retraso_60dias'] <= 12), '60Enconded'] = 3
    df.loc[(df['Nro_retraso_60dias'] > 12) & (df['Nro_retraso_60dias'] <= 15), '60Enconded'] = 4
    df.loc[ df['Nro_retraso_60dias'] > 15, '60Enconded'] = 5
    
    # MAPPING 'Nro_creditos_hipotecarios' DE LOS CLIENTES
    df.loc[ df['Nro_creditos_hipotecarios'] <= 0, 'HipEnconded'] = 0
    df.loc[(df['Nro_creditos_hipotecarios'] > 0) & (df['Nro_creditos_hipotecarios'] <= 3), 'HipEnconded'] = 1
    df.loc[(df['Nro_creditos_hipotecarios'] > 3) & (df['Nro_creditos_hipotecarios'] <= 7), 'HipEnconded'] = 2
    df.loc[(df['Nro_creditos_hipotecarios'] > 7) & (df['Nro_creditos_hipotecarios'] <= 15), 'HipEnconded'] = 3
    df.loc[(df['Nro_creditos_hipotecarios'] > 15) & (df['Nro_creditos_hipotecarios'] <= 25), 'HipEnconded'] = 4
    df.loc[ df['Nro_creditos_hipotecarios'] > 25, 'HipEnconded'] = 5
    
    # MAPPING 'Nro_dependiente' DE LOS CLIENTES
    df.loc[ df['Nro_dependiente'] <= 0, 'DepEncoded'] = 0
    df.loc[(df['Nro_dependiente'] > 0) & (df['Nro_dependiente'] <= 2), 'DepEncoded'] = 1
    df.loc[(df['Nro_dependiente'] > 2) & (df['Nro_dependiente'] <= 5), 'DepEncoded'] = 2
    df.loc[(df['Nro_dependiente'] > 5) & (df['Nro_dependiente'] <= 10), 'DepEncoded'] = 3
    df.loc[(df['Nro_dependiente'] > 10) & (df['Nro_dependiente-'] <= 13), 'DepEncoded'] = 4
    df.loc[ df['Nro_dependiente'] > 13, 'DepEncoded'] = 5
    
    # MAPPING 'Prct_uso_tc' DE LOS CLIENTES
    df.loc[ df['Prct_uso_tc'] <= 0.1, 'TCEncoded'] = 0
    df.loc[(df['Prct_uso_tc'] > 0.1) & (df['Prct_uso_tc'] <= 0.5), 'TCEncoded'] = 1
    df.loc[(df['Prct_uso_tc'] > 0.5) & (df['Prct_uso_tc'] <= 1.0), 'TCEncoded'] = 2
    df.loc[(df['Prct_uso_tc'] > 1.0) & (df['Prct_uso_tc'] <= 1.5), 'TCEncoded'] = 3
    df.loc[(df['Prct_uso_tc'] > 1.5) & (df['Prct_uso_tc'] <= 2.0), 'TCEncoded'] = 4
    df.loc[ df['Prct_uso_tc'] > 2.0, 'TCEncoded'] = 5
    
    # MAPPING 'Prct_deuda_vs_ingresos' DE LOS CLIENTES
    df.loc[ df['Prct_deuda_vs_ingresos'] <= 0.1, 'DeuInEncoded'] = 0
    df.loc[(df['Prct_deuda_vs_ingresos'] > 0.1) & (df['Prct_deuda_vs_ingresos'] <= 0.5), 'DeuInEncoded'] = 1
    df.loc[(df['Prct_deuda_vs_ingresos'] > 0.5) & (df['Prct_deuda_vs_ingresos'] <= 1.0), 'DeuInEncoded'] = 2
    df.loc[(df['Prct_deuda_vs_ingresos'] > 1.0) & (df['Prct_deuda_vs_ingresos'] <= 1.5), 'DeuInEncoded'] = 3
    df.loc[(df['Prct_deuda_vs_ingresos'] > 1.5) & (df['Prct_deuda_vs_ingresos'] <= 2.0), 'DeuInEncoded'] = 4
    df.loc[ df['Prct_deuda_vs_ingresos'] > 2.0, 'DeuInEncoded'] = 5
    
    # MAPPING 'Mto_ingreso_mensual' DE LOS CLIENTES
    df.loc[ df['Mto_ingreso_mensual'] <= 1000, 'IngresoEncoded'] = 0
    df.loc[(df['Mto_ingreso_mensual'] > 1000) & (df['Mto_ingreso_mensual'] <= 3000), 'IngresoEncoded'] = 1
    df.loc[(df['Mto_ingreso_mensual'] > 3000) & (df['Mto_ingreso_mensual'] <= 5000), 'IngresoEncoded'] = 2
    df.loc[(df['Mto_ingreso_mensual'] > 5000) & (df['Mto_ingreso_mensual'] <= 8000), 'IngresoEncoded'] = 3
    df.loc[(df['Mto_ingreso_mensual'] > 8000) & (df['Mto_ingreso_mensual'] <= 14000), 'IngresoEncoded'] = 4
    df.loc[ df['Mto_ingreso_mensual'] > 14000, 'IngresoEncoded'] = 5
    
    # MAPPING 'Nro_retraso_ultm3anios' DE LOS CLIENTES
    df.loc[ df['Nro_retraso_ultm3anios'] <= 0, 'UltiEncoded'] = 0
    df.loc[(df['Nro_retraso_ultm3anios'] > 0) & (df['Nro_retraso_ultm3anios'] <= 2), 'UltiEncoded'] = 1
    df.loc[(df['Nro_retraso_ultm3anios'] > 2) & (df['Nro_retraso_ultm3anios'] <= 5), 'UltiEncoded'] = 2
    df.loc[(df['Nro_retraso_ultm3anios'] > 5) & (df['Nro_retraso_ultm3anios'] <= 8), 'UltiEncoded'] = 3
    df.loc[(df['Nro_retraso_ultm3anios'] > 8) & (df['Nro_retraso_ultm3anios'] <= 11), 'UltiEncoded'] = 4
    df.loc[ df['Nro_retraso_ultm3anios'] > 11, 'UltiEncoded'] = 5
    
    #ELIMINAR LOS CAMPOS Y REEMPLAZAR POR LOS RANGOS DEFINIDOS
    drop_elements = ['Nro_prestao_retrasados','Nro_retraso_60dias','Nro_retraso_ultm3anios','Edad','Mto_ingreso_mensual','Nro_prod_financieros_deuda','Nro_creditos_hipotecarios','Nro_dependiente','Prct_uso_tc','Prct_deuda_vs_ingresos']
    df = df.drop(drop_elements, axis = 1)
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
    df1 = read_file_csv('2_DS_creditos.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1,
['Default','Nro_prestao_retrasados','Nro_retraso_60dias','Nro_retraso_ultm3anios','Edad','Mto_ingreso_mensual','Nro_prod_financieros_deuda','Nro_creditos_hipotecarios','Nro_dependiente','Prct_uso_tc','Prct_deuda_vs_ingresos'],'creditos_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('2_DS_creditos_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, 
['Default','Nro_prestao_retrasados','Nro_retraso_60dias','Nro_retraso_ultm3anios','Edad','Mto_ingreso_mensual','Nro_prod_financieros_deuda','Nro_creditos_hipotecarios','Nro_dependiente','Prct_uso_tc','Prct_deuda_vs_ingresos'],'creditos_test.csv')
    # Matriz de Scoring
    df3 = read_file_csv('2_DS_creditos_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, 
['Nro_prestao_retrasados','Nro_retraso_60dias','Nro_retraso_ultm3anios','Edad','Mto_ingreso_mensual','Nro_prod_financieros_deuda','Nro_creditos_hipotecarios','Nro_dependiente','Prct_uso_tc','Prct_deuda_vs_ingresos'],'creditos_score.csv')
    
if __name__ == "__main__":
    main()


