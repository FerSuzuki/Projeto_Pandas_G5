import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from datetime import date, datetime, timedelta

# Definir o caminho da pasta em que o projeto se encontra
folder_class_path = os.path.abspath(os.getcwd())

'''
Programa com o intuito de treinar ferramentas do Pandas e realizar análise de dados com diferentes bases.
Let's Code.

Dados utilizados:
- CSV Dados Demográficos 
- CSV Renda e Gastos
- CSV Bens
'''


def main():
    df_demografico, df_renda_gastos, df_bens = receber_bases()
    df_unificado = unificar_bases(df_demografico, df_renda_gastos, df_bens)
    df_relatorio_geral = criar_relatorio_geral(df_unificado)
    df_relatorio_sem_outlier = criar_relatorio_sem_outlier(df_unificado)


# ------------------------------------------------- Tarefa 1 --------------------------------------------------------- #
# Função com o objetivo de receber as bases de dados csv
def receber_bases():
    df_demografico = pd.read_csv(folder_class_path + '/1_demografico.csv', sep=';', encoding='utf-8-sig')
    df_renda_gastos = pd.read_csv(folder_class_path + '/2_renda_gastos.csv', sep=';', encoding='utf-8-sig')
    df_bens = pd.read_csv(folder_class_path + '/3_bens.csv', sep=';', encoding='utf-8-sig')
    
    return df_demografico, df_renda_gastos, df_bens


# Função para junção das bases de dados
def unificar_bases(df_demografico, df_renda_gastos, df_bens):
    # O merge é realizado de forma interna
    df_unificado = df_demografico.merge(df_renda_gastos, how='inner')
    # É preciso utilizar ambos os index como chaves para se tornar interno
    df_unificado = df_unificado.merge(df_bens, left_index=True, right_index=True)
    
    return df_unificado

# -------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Tarefa 2 --------------------------------------------------------- #

# Função para criar o relatório de variáveis quantitativas
def criar_relatorio_geral(df_unificado):
    # Receber os valores estatísticos da função describe
    df_relatorio = df_unificado.describe().T.sort_index()
    # Remover as colunas qualitativas, mas que contém números e devem ser desconsideradas
    columns_drop = ['count', 'std', '50%']
    index_drop = ['ID', 'Agricultural Household indicator', 'Electricity']
    df_relatorio.drop(index=index_drop, columns=columns_drop, inplace=True)

    # Calcular a métrica faltante da mediana
    df_relatorio['median'] = calcula_mediana(df_unificado, df_relatorio.index.values)

    # Reordenar as colunas e devolver o DataFrame
    columns_order = ['min', '25%', 'median', '75%', 'max', 'mean']
    df_relatorio = df_relatorio[columns_order]
    
    return round(df_relatorio, 3)


# Função para retornar a mediana de uma coluna quantitativa de DataFrame
def calcula_mediana(df_unificado, colunas_qualitativas):
    serie_mediana = []
    for coluna in colunas_qualitativas:
        serie_mediana.append(df_unificado[coluna].median())
    
    return serie_mediana

# -------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Tarefa 3 --------------------------------------------------------- #

# Função para remoção dos Outliers
def criar_relatorio_sem_outlier(df_unificado):
    # Resumir somente as colunas com dados numéricos
    df_unificado = df_unificado.select_dtypes(include=np.number)

    # Definir as colunas qualitativas que se utilizam de variáveis numéricas para que sejam excluídas 
    colunas_drop = ['ID', 'Agricultural Household indicator', 'Electricity']
    df_unificado.drop(columns=colunas_drop, inplace=True)

    # Cada coluna restante deve ter os seus outliers excluídos
    for coluna in df_unificado.columns:
        df_unificado.loc[:, coluna] = remover_outliers(df_unificado[coluna])
    
    # Criar o DataFrame de relatório a partir do DF de dados
    df_relatorio_sem_outlier = df_unificado.describe().T.sort_index()
    # Remover as colunas qualitativas, mas que contém números e devem ser desconsideradas
    columns_drop = ['count', 'std', '50%']

    # Calcular a métrica faltante da mediana
    df_relatorio_sem_outlier['median'] = calcula_mediana(df_unificado, df_relatorio_sem_outlier.index.values)

    # Reordenar as colunas e devolver o DataFrame
    columns_order = ['min', '25%', 'median', '75%', 'max', 'mean']
    df_relatorio_sem_outlier = df_relatorio_sem_outlier[columns_order]

    return round(df_relatorio_sem_outlier, 3)

# Função para remover os outliers da coluna em análise do DataFrame
def remover_outliers(serie_coluna):
    q1 = np.percentile(serie_coluna, 25)
    q3 = np.percentile(serie_coluna, 75)
    delta_outlier = 1.5*(q3 - q1)
    serie_coluna_sem_outlier = serie_coluna.apply(lambda x: aplicar_metodo_tuckey(q1, q3, delta_outlier, x))

    return serie_coluna_sem_outlier

# Função para aplicar o método de Tuckey e avaliar se o dado analisado é outlier ou não
def aplicar_metodo_tuckey(q1, q3, delta_outlier, valor):
    eh_outlier_menor = valor < q1 - delta_outlier 
    eh_outlier_maior = valor > q3 + delta_outlier
    if eh_outlier_menor or eh_outlier_maior:
        valor_novo = np.nan
    else:
        valor_novo = valor

    return valor_novo

# -------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Tarefa 4 --------------------------------------------------------- #

# Funçãp para reposição dos valores de Outliers com a mediana

# -------------------------------------------------------------------------------------------------------------------- #

# Definir os argumentos que o programa pode receber
parser = ArgumentParser()
parser.add_argument('-to', '--tipo_operacao', type=str,
                    help='Define o tipo de operação que será realizada: Requisição API BBCE ou Enviar os dados ao BD',
                    default='api')

if __name__ == '__main__':
    main()
