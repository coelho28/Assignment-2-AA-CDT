import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator

class FrozenTransformer(BaseEstimator):
    def __init__(self, fitted_transformer):
        self.fitted_transformer = fitted_transformer

    def __getattr__(self, name):
        return getattr(self.fitted_transformer, name)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.fitted_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fitted_transformer.transform(X)


# Função para calcular o erro censurado (cMSE)
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

def zero_missing_values(X_data):
    for column in X_data.columns:
        if X_data[column].dtype in ['float64', 'int64']:  # Verificar se a coluna é numérica
            X_data.loc[:, column] = X_data[column].fillna(0)
    return X_data

# Carregar os dados
dataTrain = pd.read_csv('C:/Users/vidcoelh/Documents/Pessoal/AA/Proj22/Proj2/train_data.csv')
dataTrain = dataTrain.drop(dataTrain.columns[0], axis=1)  # Remover a primeira coluna (id)
# Remover colunas com missing values (exceto SurvivalTime) e filtrar linhas com Censored = 1
dataTrain = dataTrain.dropna(subset=["SurvivalTime"])
dataTrain = zero_missing_values(dataTrain)

# Separar features (X) e target (y) e a variável de censura (c)
X = dataTrain.drop(columns=["SurvivalTime", "Censored"])
y = dataTrain["SurvivalTime"]
c = dataTrain["Censored"]

# Dividir em treino + validação e teste (80% para treino + validação, 20% para teste)
X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
    X, y, c, test_size=0.2, random_state=42
)

X_labeled, X_unlabeled, y_labeled, y_unlabeled, c_labeled, c_unlabeled = train_test_split(X, y, c, test_size=0.5, random_state=42)

X_combined = pd.concat([X_labeled, X_unlabeled], ignore_index=True)

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

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Train Isomap
iso = Isomap(n_components=2)
iso.fit_transform(X_scaled)

pipe = make_pipeline(SimpleImputer(),
                     scaler,
                     FrozenTransformer(iso),  # Frozen Isomap
                     GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2))

# print(X_train)
# Instanciar e treinar o modelo
#model = LinearRegression()
pipe.fit(X_labeled, y_labeled)

y_train_pred = pipe.predict(X_train)
y_val_pred = pipe.predict(X_val)

#val_cMSE = error_metric(y_val, y_val_pred, c_val)
#test_cMSE = error_metric(y_train, y_train_pred, c_train)

# Avaliar o modelo no conjunto de validação e teste usando a métrica cMSE
val_cMSE = error_metric(y_val, y_val_pred, c_val)
train_cMSE = error_metric(y_train, y_train_pred, c_train)

# Exibir os resultados
print(f"cMSE no conjunto de validação: {val_cMSE}")
print(f"cMSE no conjunto de treino: {train_cMSE}")


test = pd.read_csv('C:/Users/vidcoelh/Documents/Pessoal/AA/Proj22/Proj2/test_data.csv')
cols = ['id','Age','Gender','Stage','GeneticRisk','TreatmentType','ComorbidityIndex','TreatmentResponse']
test.columns=cols

entry_test = X.columns.tolist()
#X_test = zero_missing_values(test[entry_test])
y_test_pred = pipe.predict(test[entry_test])

#Salvar os resultados no formato solicitado
output_df_knn = pd.DataFrame(y_test_pred, columns=["SurvivalTime"])
output_df_knn['id'] = test['id']
output_df_knn = output_df_knn[['id', 'SurvivalTime']]
output_df_knn.to_csv('semisupervised-submission-02.csv', index=False)

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
plt.ylabel('Y previsto', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()