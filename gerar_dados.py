import csv
import random
from datetime import datetime, timedelta

def gerar_dados_sorveteria(qtd_dias=100, caminho_arquivo="inputs/sorvetes_vendas.csv"):
    random.seed(42)  # Garante reprodutibilidade dos dados fictícios
    
    data_inicial = datetime(2026, 1, 1)
    linhas = [["data", "temperatura_celsius", "fim_de_semana", "vendas_sorvetes"]]
    
    for i in range(qtd_dias):
        data_atual = data_inicial + timedelta(days=i)
        data_str = data_atual.strftime("%Y-%m-%d")
        
        # 1 se for sábado (5) ou domingo (6), senão 0
        eh_fim_de_semana = 1 if data_atual.weekday() in [5, 6] else 0
        
        # Temperatura entre 18.0°C e 38.0°C (cidade litorânea)
        temperatura = round(random.uniform(18.0, 38.0), 1)
        
        # Relação linear com temperatura + bônus de fim de semana + ruído aleatório
        # Cada grau acima de 18°C adiciona cerca de 18 sorvetes
        vendas_base = 100
        impacto_temperatura = (temperatura - 18.0) * 19.5
        bonus_fim_semana = 45 if eh_fim_de_semana else 0
        ruido = random.uniform(-15, 15)
        
        total_vendas = int(max(20, round(vendas_base + impacto_temperatura + bonus_fim_semana + ruido)))
        
        linhas.append([data_str, temperatura, eh_fim_de_semana, total_vendas])
        
    with open(caminho_arquivo, mode="w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerows(linhas)
        
    print(f"Dataset gerado com sucesso: {qtd_dias} registros salvos em '{caminho_arquivo}'!")

if __name__ == "__main__":
    gerar_dados_sorveteria()
