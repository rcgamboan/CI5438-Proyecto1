import pandas as pd
import os
import sklearn

import category_encoders as ce


def clean_data(file, null_data_method):
    """
    Implementa la funcion que se encargara de limpiar el archivo de data.

    Args:
        file: nombre del archivo a limpiar

    Returns:
        Matriz de data.
    """

    # Leer archivo
    df = pd.read_csv(file, sep=',')

    # Eliminar columnas que no se usaran
    df = df.drop(
        [
            'Model',
            'Location',
            'Color',
            'Seller Type',
            'Engine',
            'Max Power',
            'Max Torque',
            'Drivetrain',
            'Length',
            'Width',
            'Height'], 
        axis=1)
    
    # Eliminar filas con datos faltantes
    # Metodo 1: eliminar filas con datos faltantes
    # Metodo 2: reemplazar datos faltantes con el valor mas comun en la columna
    if null_data_method == 1:
        df = df.dropna()
    elif null_data_method == 2:
        df = df.fillna(df.mode().iloc[0])

    # Categorizacion de la data
    # Usando la libreria sklearn con one-hot encoding
    ohe = ce.OneHotEncoder(cols=['Make',
                                 'Fuel Type',
                                 'Transmission',
                                 'Owner'])
    df = ohe.fit_transform(df)

    # Guardamos la lista de precios sin normalizar
    precios_sin_normalizar = df['Price']

    # Normalizacion de la data con valores numericos altos
    columns_to_normalize = ['Price',
                            'Year',
                            'Kilometer',
                            'Seating Capacity',
                            'Fuel Tank Capacity']
    
    convert_dict = {'Price': float,
                    'Year': float,
                    'Kilometer': float,
                    'Seating Capacity': float,
                    'Fuel Tank Capacity': float}
    df = df.astype(convert_dict)
    
    for column in columns_to_normalize:
        max_value = df[column].max()
        min_value = df[column].min()
        for index, row in df.iterrows():
            df.at[index, column] = float((row[column] - min_value) / (max_value - min_value))


    # Guardar data limpia en un nuevo archivo
    # con el mismo nombre agregando '_clean' al final
    filename, file_extension = os.path.splitext(file)
    ruta = filename + '_clean' + file_extension
    
    # Guardar data limpia en un nuevo archivo
    df.to_csv(ruta, index=False)

    return df, precios_sin_normalizar

