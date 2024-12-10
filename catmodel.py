import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Função para calcular o erro censurado (cMSE)
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

# Carregar os dados
dataTrain = pd.read_csv('./train_data.csv')
dataTrain = dataTrain.drop(dataTrain.columns[0], axis=1)  # Remover a primeira coluna (id)

# Remover colunas com missing values (exceto SurvivalTime) e filtrar linhas com Censored = 1
dataTrain = dataTrain.dropna(subset=["SurvivalTime"])

# Separar features (X) e target (y) e a variável de censura (c)
X = dataTrain.drop(columns=["SurvivalTime", "Censored"])
y = dataTrain["SurvivalTime"]
c = dataTrain["Censored"]

# Dividir em treino + validação e teste (80% para treino + validação, 20% para teste)
X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(X, y, c, test_size=0.2, random_state=42)

# Definir o modelo base para o GridSearchCV
model = CatBoostRegressor(cat_features=[], verbose=0)

# Definir a estratégia de busca de hiperparâmetros
param_grid = {
    'iterations': [160, 170, 180],               # Testando diferentes números de iterações
    'learning_rate': [0.05],            # Testando diferentes taxas de aprendizagem
    'depth': [1],                          # Testando diferentes profundidades de árvores
    'l2_leaf_reg': [5],                     # Testando diferentes valores de regularização L2
    'bagging_temperature': [0.2],         # Testando diferentes valores de bagging temperature
    'random_strength': [10]               # Testando diferentes forças de aleatoriedade
}

# Instanciar o GridSearchCV com verbose=2 para feedback detalhado
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Treinar o modelo com GridSearchCV
grid_search.fit(X_train, y_train, sample_weight=(1 - c_train))

# Exibir os melhores parâmetros encontrados
print("Melhores parâmetros encontrados:", grid_search.best_params_)

# Usar o melhor modelo encontrado para fazer previsões
best_model = grid_search.best_estimator_

# Predições no conjunto de validação e treino
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)

# Calcular o erro censurado (cMSE)
train_cMSE = error_metric(y_train, y_train_pred, c_train)
val_cMSE = error_metric(y_val, y_val_pred, c_val)

# Exibir os resultados
print(f"cMSE no conjunto de validação: {val_cMSE}")

# Previsões para os dados de teste
test = pd.read_csv('./test_data.csv')
cols = ['id','Age','Gender','Stage','GeneticRisk','TreatmentType','ComorbidityIndex','TreatmentResponse']
test.columns = cols

entry_test = X.columns.tolist()
X_test = test[entry_test]

# Predições do modelo para o conjunto de teste
y_test_pred = best_model.predict(X_test)

# Salvar os resultados no formato solicitado
output_df_knn = pd.DataFrame(y_test_pred, columns=["SurvivalTime"])
output_df_knn['id'] = test['id']
output_df_knn = output_df_knn[['id', 'SurvivalTime']]
output_df_knn.to_csv('handle-missing-submission-04.csv', index=False)

# Visualização dos resultados
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
