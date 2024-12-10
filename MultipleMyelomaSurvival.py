import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados e aplicar a filtragem conforme necessário
dataTrain = pd.read_csv('./train_data.csv')
cols_with_missing_points = [col for col in dataTrain.columns if dataTrain[col].isnull().any() and col != 'SurvivalTime']
dataTrain = dataTrain.drop(columns=cols_with_missing_points)
# dataTrain = dataTrain[dataTrain["Censored"] != 1]
dataTrain = dataTrain.dropna(subset=["SurvivalTime"])
dataTrain = dataTrain.drop(dataTrain.columns[0], axis=1)  # Remover a primeira coluna (id)

# Separar features (X), target (y) e variável de censura (c)
X = dataTrain.drop(columns=["SurvivalTime", "Censored"])
y = dataTrain["SurvivalTime"]
c = dataTrain["Censored"]

# Dividir em treino e teste (80% para treino, 20% para teste)
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
    X, y, c, test_size=0.2, random_state=42
)

# Função de métrica de erro censurado (cMSE)
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

def derivative_error_metric(y, y_hat, c):
    err = y - y_hat
    # Case 1: err >= 0
    grad_positive = -2 * err
    
    # Case 2: err < 0
    grad_negative = -2 * (1 - c) * err
    
    # Combine cases based on the condition
    grad = np.where(err >= 0, grad_positive, grad_negative)
    
    # Normalize by the number of samples
    grad /= err.shape[0]
    # print(grad)
    return grad


def gradient_descent(X, y, y_hat, c, learning_rate=0.05, iterations=100):
    # Inicializa os parâmetros (por exemplo, pesos)
    params = np.random.randn(X.shape[1])  # Assume que X tem as features do modelo
    for i in range(iterations):
        # Calcula o gradiente da função de erro (cMSE) em relação aos parâmetros
        gradient = derivative_error_metric(y, y_hat, c)
        
        # Atualiza os parâmetros com base no gradiente e taxa de aprendizado
        params = params - learning_rate * gradient
        
        # Pode-se calcular a função de erro para monitorar o progresso (opcional)
        if i % 10 == 0:  # Imprime a cada 10 iterações
            error = error_metric(y, y_hat, c)
            print(f'Iteração {i}, Erro: {error}')
    
    return params

pipeline = Pipeline([('scaler', StandardScaler()),
             ('linear_regression', LinearRegression())])
# Configurar a validação cruzada no conjunto de treino
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_errors = []

# Realizar validação cruzada
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    c_train_fold, c_val_fold = c_train.iloc[train_index], c_train.iloc[val_index]

    pipeline.fit(X_train_fold, y_train_fold)
    # Treinar o modelo no fold atual
    y_val_pred = pipeline.predict(X_val_fold)
    y_train_pred = pipeline.predict(X_train)
    # Avaliar o modelo usando cMSE e armazenar o resultado
    fold_cMSE = derivative_error_metric(y_val_fold, y_val_pred, c_val_fold)
    cross_val_errors.append(fold_cMSE)

# print(cross_val_errors)
# Calcular a média do erro de validação cruzada
all_values = np.concatenate(cross_val_errors)
mean_cross_val_error = np.mean(all_values)
print(f"Média do cMSE nos folds de validação cruzada: {mean_cross_val_error}")
# Treinar o modelo no conjunto completo de treino

# Fazer previsões no conjunto de teste e avaliar




y_test_pred = pipeline.predict(X_test)

test_cMSE = derivative_error_metric(y_test, y_test_pred, c_test)
mean_cross_val_error_test = np.mean(test_cMSE)
train_cMSE = derivative_error_metric(y_train, y_train_pred, c_train)
mean_cross_val_error_train = np.mean(train_cMSE)
print(f"Derivada de cMSE no conjunto de test: {mean_cross_val_error_test}")
print(f"Derivada de cMSE no conjunto de train: {mean_cross_val_error_train}")
test_cMSE = error_metric(y_test, y_test_pred, c_test)
train_cMSE = error_metric(y_train, y_train_pred, c_train)
print(f"cMSE no conjunto de teste: {test_cMSE}")

test = pd.read_csv('./test_data.csv')
cols = ['id','Age','Gender','Stage','GeneticRisk','TreatmentType','ComorbidityIndex','TreatmentResponse']
test.columns=cols

entry_test = X_train.columns.tolist()
columns = ['SurvivalTime']
# Exportar para um arquivo CSV
output_df = pd.DataFrame( pipeline.predict(test[entry_test]), columns=columns)
output_df['id'] = output_df.index
output_df = output_df[['id', 'SurvivalTime']]
output_df.to_csv('baseline-submission-04.csv', index=False)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', label='Train', alpha=0.5)
plt.scatter(y_test, y_test_pred, color='red', label='Test', alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='black', linestyle='--')  # Linha de perfeição
plt.xlabel('Real Survival Time')
plt.ylabel('Predicted Survival Time')
plt.legend()

plt.tight_layout()
plt.show()