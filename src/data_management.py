import pandas as pd


def clean_data(file):
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
            'Price',
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
    df = df.dropna()

    # Metodo 2: reemplazar datos faltantes con la media
    # for columnName in df:
    #     column_mean = df[columnName].mean()
    #     df[columnName].fillna(value=column_mean, inplace=True)

    # Metodo 3: reemplazar datos faltantes con valor mas comun de la columna
    # df = df.fillna(df.mode().iloc[0])

    df.to_csv('doc/CarDekho_clean.csv', index=False)
    return df

def main():
    clean_data('doc/CarDekho.csv')

if __name__ == "__main__":
    main()