import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

# Carregar dados
dataTrain = pd.read_csv('./train_data.csv')

# Remover colunas com valores ausentes, exceto SurvivalTime e Censored
cols_with_missing_points = [col for col in dataTrain.columns if dataTrain[col].isnull().any() and col != 'SurvivalTime' and col != 'Censored']
dataTrain = dataTrain.drop(columns=cols_with_missing_points)
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
# Função de validação cruzada para regressão polinomial
def validate_polynomial_with_gradients(X, y, c, degree_values=range(1, 6), alpha_values=[0.1, 1.0, 10.0]):
    best_cMSE = float('inf')
    best_model = None
    best_params = None

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for degree in degree_values:
        for alpha in alpha_values:
            print(f"Validating polynomial regression with degree={degree}, alpha={alpha}...")

            # Criar pipeline com polinomial e regressão ridge
            model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=alpha))

            fold_errors_val = []
            fold_gradients = []

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

                # Calcular os gradientes
                gradients = error_metric_derivative(y_val_fold, y_val_pred, c_val_fold)
                mean_gradient_magnitude = np.mean(np.abs(gradients))
                fold_gradients.append(mean_gradient_magnitude)

            # Calcular métricas agregadas dos gradientes
            mean_cMSE_val = np.mean(fold_errors_val)
            mean_gradient_across_folds = np.mean(fold_gradients)

            print(f"Degree {degree}, Alpha {alpha}: Mean cMSE = {mean_cMSE_val}, Mean Gradient Magnitude = {mean_gradient_across_folds}")

            # Atualizar o melhor modelo
            if mean_cMSE_val < best_cMSE:
                best_cMSE = mean_cMSE_val
                best_model = model
                best_params = (degree, alpha)

    print(f"Best Parameters: Degree={best_params[0]}, Alpha={best_params[1]}, Best cMSE={best_cMSE}")
    return best_model, best_cMSE, best_params


# Treinar o modelo polinomial usando validação cruzada
best_model, best_cMSE, best_params = validate_polynomial_with_gradients(X, y, c)

# Previsões no conjunto de treino (para visualização)
y_train_pred = best_model.predict(X_train)

# Função para criar o gráfico de y vs y_hat
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k', label='Predicted Points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Values (y)")
plt.ylabel("Predicted Values (y_hat)")
plt.title("Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()

# Previsões para o conjunto de teste
test = pd.read_csv('./test_data.csv')
cols = ['id', 'Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']
test.columns = cols

# As colunas de entrada devem ser as mesmas do conjunto de treino
entry_test = X.columns.tolist()
y_test_pred_poly = best_model.predict(test[entry_test])

# Salvar os resultados no formato solicitado
output_df_poly = pd.DataFrame(y_test_pred_poly, columns=["SurvivalTime"])
output_df_poly['id'] = test['id']
output_df_poly = output_df_poly[['id', 'SurvivalTime']]
output_df_poly.to_csv('Polynomial-submission.csv', index=False)
