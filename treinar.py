import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def carregar_dados(caminho_csv="inputs/sorvetes_vendas.csv"):
    """Carrega o dataset e separa features (X) e target (y)."""
    df = pd.read_csv(caminho_csv)
    
    # Features preditivas (Temperatura e Fim de semana)
    X = df[["temperatura_celsius", "fim_de_semana"]]
    
    # Target (Vendas de sorvetes a prever)
    y = df["vendas_sorvetes"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def avaliar_e_registrar_modelo(nome_modelo, modelo, X_train, X_test, y_train, y_test):
    """Treina, avalia e registra o modelo e suas métricas no MLflow."""
    with mlflow.start_run(run_name=nome_modelo):
        # 1. Treinamento
        modelo.fit(X_train, y_train)
        
        # 2. Previsões no conjunto de teste
        y_pred = modelo.predict(X_test)
        
        # 3. Métricas de avaliação
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 4. Rastreamento com MLflow (Parâmetros, Métricas e Artefatos)
        mlflow.log_param("algoritmo", nome_modelo)
        mlflow.log_param("qtd_treino", len(X_train))
        mlflow.log_param("qtd_teste", len(X_test))
        
        mlflow.log_metric("mae", round(mae, 2))
        mlflow.log_metric("rmse", round(rmse, 2))
        mlflow.log_metric("r2_score", round(r2, 4))
        
        # Registra o modelo treinado
        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="modelo_sorvetes"
        )
        
        print(f"\n--- {nome_modelo} ---")
        print(f"  MAE (Erro Médio Absoluto): {mae:.2f} unidades")
        print(f"  RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f} unidades")
        print(f"  R² Score (Acurácia explicada da variância): {r2:.4f} ({r2*100:.2f}%)")

def main():
    print("Iniciando pipeline de Machine Learning com MLflow...")
    
    # Define o experimento no MLflow (equivalente aos Jobs do Azure ML)
    mlflow.set_experiment("Previsao_Vendas_Gelato_Magico")
    
    X_train, X_test, y_train, y_test = carregar_dados()
    
    # Modelo 1: Regressão Linear (modelo clássico de base)
    modelo_linear = LinearRegression()
    avaliar_e_registrar_modelo("Regressao_Linear", modelo_linear, X_train, X_test, y_train, y_test)
    
    # Modelo 2: Random Forest Regressor (para comparar experimentos no MLflow)
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    avaliar_e_registrar_modelo("Random_Forest_Regressor", modelo_rf, X_train, X_test, y_train, y_test)
    
    print("\nTreinamento concluído e registrado no MLflow com sucesso!")
    print("Para visualizar o dashboard de experimentos, execute no terminal:")
    print("  mlflow ui")

if __name__ == "__main__":
    main()
