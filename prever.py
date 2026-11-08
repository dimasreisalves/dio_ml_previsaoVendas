import mlflow
import pandas as pd
import sys

def carregar_modelo_mais_recente():
    """Busca o último modelo salvo no experimento do MLflow."""
    client = mlflow.tracking.MlflowClient()
    experimento = client.get_experiment_by_name("Previsao_Vendas_Gelato_Magico")
    
    if not experimento:
        print("Experimento não encontrado. Execute primeiro: python treinar.py")
        sys.exit(1)
        
    runs = client.search_runs(
        experiment_ids=[experimento.experiment_id],
        order_by=["metrics.r2_score DESC"],  # Pega o melhor modelo por R²
        max_results=1
    )
    
    if not runs:
        print("Nenhuma execução encontrada. Execute primeiro: python treinar.py")
        sys.exit(1)
        
    melhor_run = runs[0]
    run_id = melhor_run.info.run_id
    nome_modelo = melhor_run.data.params.get("algoritmo", "Modelo")
    r2 = melhor_run.data.metrics.get("r2_score", 0)
    
    print(f"Carregando o melhor modelo: {nome_modelo} (Run ID: {run_id[:8]}... | R²: {r2})")
    
    modelo_uri = f"runs:/{run_id}/modelo_sorvetes"
    return mlflow.pyfunc.load_model(modelo_uri)

def prever_vendas(modelo, temperatura, eh_fim_de_semana):
    """Realiza a previsão em tempo real para uma dada temperatura."""
    dados_entrada = pd.DataFrame([{
        "temperatura_celsius": float(temperatura),
        "fim_de_semana": int(eh_fim_de_semana)
    }])
    
    previsao = modelo.predict(dados_entrada)
    return round(float(previsao[0]))

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    modelo = carregar_modelo_mais_recente()
    
    print("\n=======================================================")
    print("[Gelato Magico] - Sistema de Previsao de Vendas (ML)")
    print("=======================================================\n")
    
    # Simulacoes de teste em tempo real:
    testes = [
        {"temp": 20.0, "fds": 0, "desc": "Dia ameno durante a semana"},
        {"temp": 28.5, "fds": 0, "desc": "Dia quente durante a semana"},
        {"temp": 33.0, "fds": 1, "desc": "Sabado ensolarado e muito quente"},
        {"temp": 38.0, "fds": 1, "desc": "Domingo de calor extremo"},
    ]
    
    print("Previsoes em tempo real geradas pelo modelo:")
    for t in testes:
        qtd_prevista = prever_vendas(modelo, t["temp"], t["fds"])
        fds_texto = "Sim" if t["fds"] == 1 else "Nao"
        print(f"  * {t['desc']} ({t['temp']} C | Fim de semana: {fds_texto}) => Previsao: {qtd_prevista} sorvetes")
    print("\nPronto para planejar a producao do dia e evitar desperdicios!")
