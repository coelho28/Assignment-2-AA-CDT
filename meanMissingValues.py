import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Função para calcular o erro censurado (cMSE)
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

def mean_missing_values(X_data):
    for column in X_data.columns:
        if X_data[column].dtype in ['float64', 'int64']:  # Verificar se a coluna é numérica
            mean_value = X_data[column].mean(skipna=True)
            X_data.loc[:, column] = X_data[column].fillna(mean_value)
    return X_data
    
    

# Carregar os dados
dataTrain = pd.read_csv('./train_data.csv')
dataTrain = dataTrain.drop(dataTrain.columns[0], axis=1)  # Remover a primeira coluna (id)
# Remover colunas com missing values (exceto SurvivalTime) e filtrar linhas com Censored = 1
dataTrain = dataTrain.dropna(subset=["SurvivalTime"])
dataTrain = mean_missing_values(dataTrain)

# Separar features (X) e target (y) e a variável de censura (c)
X = dataTrain.drop(columns=["SurvivalTime", "Censored"])
y = dataTrain["SurvivalTime"]
c = dataTrain["Censored"]

# Dividir em treino + validação e teste (80% para treino + validação, 20% para teste)
X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
    X, y, c, test_size=0.2, random_state=42
)

# print(X_train)
# Instanciar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

val_cMSE = error_metric(y_val, y_val_pred, c_val)
test_cMSE = error_metric(y_train, y_train_pred, c_train)

# Avaliar o modelo no conjunto de validação e teste usando a métrica cMSE
val_cMSE = error_metric(y_val, y_val_pred, c_val)
train_cMSE = error_metric(y_train, y_train_pred, c_train)

# Exibir os resultados
print(f"cMSE no conjunto de validação: {val_cMSE}")
print(f"cMSE no conjunto de teino: {train_cMSE}")


test = pd.read_csv('./test_data.csv')
cols = ['id','Age','Gender','Stage','GeneticRisk','TreatmentType','ComorbidityIndex','TreatmentResponse']
test.columns=cols

entry_test = X.columns.tolist()
X_test = mean_missing_values(test[entry_test])
y_test_pred = model.predict(X_test)

# Salvar os resultados no formato solicitado
output_df_knn = pd.DataFrame(y_test_pred, columns=["SurvivalTime"])
output_df_knn['id'] = test['id']
output_df_knn = output_df_knn[['id', 'SurvivalTime']]
output_df_knn.to_csv('baseline-submission-08.csv', index=False)

plt.figure(figsize=(10, 8))

# Treino: valores reais vs. previstos
plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k', color='blue', label='Treino')

# Validação: valores reais vs. previstos
plt.scatter(y_val, y_val_pred, alpha=0.6, edgecolor='k', color='orange', label='Validação')

# Linha de referência (Y = Y_hat)
plt.plot([min(y_train.min(), y_val.min()), max(y_train.max(), y_val.max())],
         [min(y_train.min(), y_val.min()), max(y_train.max(), y_val.max())],
         '--r', linewidth=2, label='Linha ideal (Y = Y_hat)')

# Personalização do gráfico
plt.title('Gráfico Y vs. Y_hat (Treino e Validação)', fontsize=16)
plt.xlabel('Y real', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()