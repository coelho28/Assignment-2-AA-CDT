import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

# Carregar dados
dataTrain = pd.read_csv('./train_data.csv')

# Remover colunas com valores ausentes, exceto SurvivalTime e Censored
cols_with_missing_points = [col for col in dataTrain.columns if dataTrain[col].isnull().any() and col != 'SurvivalTime' and col != 'Censored']
# dataTrain = dataTrain.drop(columns=cols_with_missing_points)
# dataTrain = dataTrain[dataTrain["Censored"] != 1]
# Remover linhas com SurvivalTime ausente
dataTrain = dataTrain.dropna(subset=["SurvivalTime"])

# Remover a primeira coluna (ID)
dataTrain = dataTrain.drop(dataTrain.columns[0], axis=1)

# Separar features, target e censoring
X = dataTrain.drop(columns=["SurvivalTime", "Censored"])
y = dataTrain["SurvivalTime"]
c = dataTrain["Censored"]

X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
    X, y, c, test_size=0.2, random_state=42
)

# Função de erro para dados censored
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]
def error_metric_derivative(y, y_hat, c):
    err = y - y_hat
    gradient = -2 * (1 - c) * err - 2 * c * (err > 0) * err
    return gradient / len(y)
# Função de validação cruzada para k-NN

def validate_knn(X, y, c, n_neighbors_values=range(1, 50)):
    best_cMSE = float('inf')
    best_model = None
    best_n_neighbors = None

    kf = KFold(n_splits=20, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    for n_neighbors in n_neighbors_values:
        # Criar o regressor k-NN
        regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
        model = make_pipeline(StandardScaler(), regressor)

        fold_errors_val = []

        for train_index, val_index in kf.split(X):
            # Separar os dados de treino e validação
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            c_train_fold, c_val_fold = c.iloc[train_index], c.iloc[val_index]

            # Treinar o modelo
            model.fit(X_train_fold, y_train_fold)
            
            # Prever no conjunto de validação
            y_val_pred = model.predict(X_val_fold)
            # Calcular o erro usando a métrica customizada
            fold_error_val = error_metric(y_val_fold, y_val_pred, c_val_fold)
            fold_errors_val.append(fold_error_val)

        # Calcular a média dos erros nos folds
        mean_cMSE_val = np.mean(fold_errors_val)
        print(f"Val n_neighbors {n_neighbors}: Mean RMSE = {mean_cMSE_val}")
        mean_cmse = np.mean(fold_errors_val)

        # Atualizar o melhor modelo
        if mean_cmse < best_cMSE:
            best_cMSE = mean_cmse
            best_model = model
            best_n_neighbors = n_neighbors

    print(f"Best n_neighbors: {best_n_neighbors}, Best RMSE: {best_cMSE}")
    return best_model, best_cMSE, best_n_neighbors


# Treinar o modelo k-NN usando validação cruzada
best_model, best_cMSE, best_n_neighbors = validate_knn(X, y, c)

# Previsões no conjunto de treino (para visualização)
y_train_pred = best_model.predict(X_train)

# Plotar y vs y_hat para o conjunto de treino

# Previsões para o conjunto de teste
test = pd.read_csv('./test_data.csv')
cols = ['id', 'Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']
test.columns = cols

# As colunas de entrada devem ser as mesmas do conjunto de treino
entry_test = X.columns.tolist()
y_test_pred_knn = best_model.predict(test[entry_test])

# Salvar os resultados no formato solicitado
output_df_knn = pd.DataFrame(y_test_pred_knn, columns=["SurvivalTime"])
output_df_knn['id'] = test['id']
output_df_knn = output_df_knn[['id', 'SurvivalTime']]
output_df_knn.to_csv('Nonlinear-submission-04.csv', index=False)

# Função para criar o gráfico de y vs y_hat
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k', label='Predicted Points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Values (y)")
plt.ylabel("Predicted Values (y_hat)")
plt.title("KNN")
plt.legend()
plt.grid(True)
plt.show()