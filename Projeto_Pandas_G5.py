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
    # Tarefa 1
    pd.options.mode.chained_assignment = None
    df_demografico, df_renda_gastos, df_bens = receber_bases()
    df_unificado = unificar_bases(df_demografico, df_renda_gastos, df_bens)
    
    # Tarefa 2
    df_relatorio_geral = criar_relatorio_geral(df_unificado)
    
    # Tarefa 3 e 4
    df_relatorio_sem_outlier = criar_relatorio_sem_outlier(df_unificado)
    df_relatorio_outlier_tratado = criar_relatorio_sem_outlier(df_unificado, opcao_tratar_outlier='tratar')
    
    # Tarefa 5
    criar_relatorio_colunas_qualitativas(df_unificado)
    
    # Tarefa 6
    cria_csv_alta_renda(df_unificado)
    pd.options.mode.chained_assignment = 'warn'


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

# ----------------------------------------------- Tarefa 3 e 4 ------------------------------------------------------- #

# Função para remoção dos Outliers
def criar_relatorio_sem_outlier(df_unificado, opcao_tratar_outlier='remover'):
    # Definir a séire de medianas do DataFrame
    median = df_unificado.median()

    # Resumir somente as colunas com dados numéricos
    df_unificado = df_unificado.select_dtypes(include=np.number)

    # Definir as colunas qualitativas que se utilizam de variáveis numéricas para que sejam excluídas 
    colunas_drop = ['ID', 'Agricultural Household indicator', 'Electricity']
    df_unificado.drop(columns=colunas_drop, inplace=True)

    # Agora, para as colunas restantes é preciso tratar os outliers (remover ou trocar pela mediana)
    for coluna in df_unificado.columns:
        df_unificado.loc[:, coluna] = tratar_outliers(df_unificado[coluna])
    
    # Para a opção de resolver os outliers com a mediana em vez de deixar vazio
    if opcao_tratar_outlier != 'remover':
        for column in df_unificado.columns:
            df_unificado[column].fillna(median[column],inplace=True)
   
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

# Função para tratar os outliers da coluna em análise do DataFrame
def tratar_outliers(serie_coluna):
    q1 = np.percentile(serie_coluna, 25)
    q3 = np.percentile(serie_coluna, 75)
    delta_outlier = 1.5*(q3 - q1)
    
    serie_coluna_outlier_tratado = serie_coluna.apply(lambda x: aplicar_metodo_tuckey(q1, q3, delta_outlier, x))

    return serie_coluna_outlier_tratado

# Função para aplicar o método de Tuckey e avaliar se o dado analisado é outlier ou não
def aplicar_metodo_tuckey(q1, q3, delta_outlier, valor):
    # Cálculo de outlier através do Método Tuckey
    eh_outlier_menor = valor < q1 - delta_outlier 
    eh_outlier_maior = valor > q3 + delta_outlier
    
    # Se o valor for outlier é necessário deixar o valor vazio
    if eh_outlier_menor or eh_outlier_maior:
        valor_novo = np.nan
    else:
        # Caso do valor não ser outlier
        valor_novo = valor

    return valor_novo
    
# -------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Tarefa 5 --------------------------------------------------------- #

# Função para criar um relatório para as variáveis qualitativas
def criar_relatorio_colunas_qualitativas(df_unificado):
    shapes = df_unificado.shape
    colunas_qualitativas = pegar_colunas_qualitativas(df_unificado)

    # Para cada coluna de variáveis qualitativas será criado um DataFrame para printar o seu respectivo relatório
    dicionario_dados = {}
    for coluna in colunas_qualitativas:
        # Para cada variável única na coluna em estudo é preciso calcular suas frequências
        for variavel_quali in sorted(df_unificado[coluna].unique()):
            # Função para retornar a lista das freqs calculadas
            dicionario_dados[variavel_quali] = calcular_frequencia(df_unificado, coluna, variavel_quali, shapes[0])
        # Construção do Data Frame a partir de um dicionário de daods
        df_temp = pd.DataFrame.from_dict(dicionario_dados, orient='index', columns=['Freq. Abs.', 'Freq. Rel.', 'Freq. Rel (%)'])
        # Construir a última coluna de freq acumulada
        df_temp.loc[:, 'Freq. Rel. Ac.'] = df_temp['Freq. Rel (%)'].cumsum()
        # Printar o DataFrame
        print(coluna)
        display(round(df_temp, 2))
        dicionario_dados = {}


# Função para pegar as colunas com dados qualitativos
def pegar_colunas_qualitativas(df_unificado):
    df_unificado.dropna(inplace=True)
    colunas_qualitativas = [coluna for coluna in df_unificado.select_dtypes(include='object').columns]
    # Colunas numéricas que representam variáveis qualitativas
    colunas_qualitativas.extend(['Agricultural Household indicator', 'Electricity'])
    colunas_qualitativas.sort()

    return colunas_qualitativas

# Função para calcular os valores de frequência de cada variável qualitativa
def calcular_frequencia(df_unificado, coluna, variavel_quali, numero_linhas):
    # Filtrar a coluna por cada variável única da sua série
    mascara_variavel_qualitativa = df_unificado[coluna] == variavel_quali
    # Calcular as frequências
    freq_abs = df_unificado.loc[mascara_variavel_qualitativa, coluna].count()
    freq_rel = freq_abs/numero_linhas

    return [freq_abs, freq_rel, freq_rel*100]


# -------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Tarefa 6 --------------------------------------------------------- #

# Função para criar um arquivo csv que resume somente os dados que a renda esteja entre os 10% maiores da base
def cria_csv_alta_renda(df_unificado):
    quantil_90 = df_unificado['Total Household Income'].quantile(0.9)
    mascara_alta_renda = df_unificado['Total Household Income'] > quantil_90

    df_alta_renda = df_unificado.loc[mascara_alta_renda].sort_values(by='Total Household Income', ascending=False)
    df_alta_renda.reset_index(drop=True,inplace=True)
    df_alta_renda.to_csv(folder_class_path + '/dados_alta_renda.csv', sep=';', encoding='utf-8-sig')


# -------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Tarefa 7 --------------------------------------------------------- #





# Definir os argumentos que o programa pode receber
parser = ArgumentParser()
parser.add_argument('-to', '--tipo_operacao', type=str,
                    help='Define o tipo de operação que será realizada: Requisição API BBCE ou Enviar os dados ao BD',
                    default='api')

if __name__ == '__main__':
    main()
