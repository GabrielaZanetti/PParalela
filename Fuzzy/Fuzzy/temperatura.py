import tkinter as tk
from tkinter import messagebox
import requests
from geopy.geocoders import Nominatim
import numpy as np
import skfuzzy as fuzz
import pandas as pd

# Coordenadas via OpenStreetMap
def obter_coordenadas(cidade):
    geolocator = Nominatim(user_agent="clima_fuzzy")
    location = geolocator.geocode(cidade)
    if location:
        return location.latitude, location.longitude
    else:
        print("Cidade não encontrada.")
        return None, None

# Dados climáticos via Open-Meteo
def obter_dados_climaticos(cidade, lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current_weather=true"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    resposta = requests.get(url)
    dados = resposta.json()
    
    if "hourly" in dados:
        temperatura = dados["hourly"]["temperature_2m"][0]
        umidade = dados["hourly"]["relative_humidity_2m"][0]
        vento = dados["hourly"]["wind_speed_10m"][0]
        
        return {"temperature": temperatura, "humidity": umidade, "wind_speed": vento}
    else:
        print("Erro ao obter os dados do clima.")
        return None

def ler_recomendacoes():
    recomendacoes = pd.read_excel('recomendacoes_atividades.xlsx')
    return recomendacoes

temperatura = np.arange(-5, 46, 1)
grau_muito_frio   = fuzz.trimf(temperatura, [-5, -5, 5])      # -5°C a 5°C maior pertinência em -5°C
grau_frio         = fuzz.trimf(temperatura, [0,  8, 16])      # 0°C a 16°C maior pertinência em 8°C
grau_medio        = fuzz.trimf(temperatura, [12, 20, 28])     # 12°C a 28°C maior pertinência em 20°C
grau_quente       = fuzz.trimf(temperatura, [24, 30, 36])     # 24°C a 36°C maior pertinência em 30°C
grau_muito_quente = fuzz.trimf(temperatura, [32, 40, 45])     # 32°C a 45°C maior pertinência em 45°C

umidade = np.arange(0, 101, 1)
umidade_muito_baixa = fuzz.trimf(umidade, [0, 0, 25])         # maior pertinência em 0% a 25% - com grau max 0
umidade_baixa       = fuzz.trimf(umidade, [10, 30, 50])       # maior pertinência em 30%
umidade_moderada    = fuzz.trimf(umidade, [30, 50, 70])       # maior pertinência por volta de 40–60%.
umidade_alta        = fuzz.trimf(umidade, [50, 70, 90])       # maior pertinência 50% a 90%.
umidade_muito_alta  = fuzz.trimf(umidade, [75, 100, 100])     # maior pertinência 75% a 100%.

vento = np.arange(0, 101, 1)
vento_fraco    = fuzz.trimf(vento, [0, 0, 15])                # 0 a 15 km/h
vento_moderado = fuzz.trimf(vento, [5, 20, 35])               # 15 a 35 km/h
vento_forte    = fuzz.trimf(vento, [25, 50, 100])             # 35 a 100 km/h

# Saída fuzzy: conforto para exercício (0-100) ---
conforto = np.arange(0, 101, 1)
conforto_nao = fuzz.trimf(conforto, [0, 0, 50])
conforto_medio = fuzz.trimf(conforto, [25, 50, 75])
conforto_sim = fuzz.trimf(conforto, [50, 100, 100])

def memberships_temperatura(temp_val):
    # retorna graus de pertinência numéricos (0..1)
    return {
        'muito_frio': fuzz.interp_membership(temperatura, grau_muito_frio, temp_val),
        'frio':       fuzz.interp_membership(temperatura, grau_frio, temp_val),
        'medio':      fuzz.interp_membership(temperatura, grau_medio, temp_val),
        'quente':     fuzz.interp_membership(temperatura, grau_quente, temp_val),
        'muito_quente': fuzz.interp_membership(temperatura, grau_muito_quente, temp_val)
    }

def memberships_umidade(umid_val):
    return {
        'muito_baixa': fuzz.interp_membership(umidade, umidade_muito_baixa, umid_val),
        'baixa':       fuzz.interp_membership(umidade, umidade_baixa, umid_val),
        'moderada':    fuzz.interp_membership(umidade, umidade_moderada, umid_val),
        'alta':        fuzz.interp_membership(umidade, umidade_alta, umid_val),
        'muito_alta':  fuzz.interp_membership(umidade, umidade_muito_alta, umid_val)
    }

def memberships_vento(vento_val):
    return {
        'fraco':    fuzz.interp_membership(vento, vento_fraco, vento_val),
        'moderado': fuzz.interp_membership(vento, vento_moderado, vento_val),
        'forte':    fuzz.interp_membership(vento, vento_forte, vento_val)
    }

def avaliar_conforto(temp_val, umid_val, vento_val):
    # Avalia  o quanto é confortável fazer exercício nas condições dadas.
    
    mt = memberships_temperatura(temp_val)
    mu = memberships_umidade(umid_val)
    mv = memberships_vento(vento_val)

    agregada = np.zeros_like(conforto, dtype=float)

    def aplicar(ativacao, out_mf):
        nonlocal agregada
        cortado = np.fmin(ativacao, out_mf)
        agregada = np.fmax(agregada, cortado)

    # Regras (exemplos simples, fáceis de ajustar):
    # 1: temperatura média + umidade moderada + vento fraco -> confortável
    aplicar(min(mt['medio'], mu['moderada'], mv['fraco']), conforto_sim)
    # 2: temperatura média + umidade baixa + vento moderado -> confortável
    aplicar(min(mt['medio'], mu['baixa'], mv['moderado']), conforto_sim)
    # 3: temperatura quente + umidade baixa + vento fraco -> confortável
    aplicar(min(mt['quente'], mu['baixa'], mv['fraco']), conforto_sim)
    # 4: temperatura média + umidade moderada + vento moderado -> médio
    aplicar(min(mt['medio'], mu['moderada'], mv['moderado']), conforto_medio)
    # 5: muito frio ou frio -> não confortável
    aplicar(max(mt['frio'], mt['muito_frio']), conforto_nao)
    # 6: muito quente -> não confortável
    aplicar(mt['muito_quente'], conforto_nao)
    # 7: umidade muito alta ou muito baixa -> não confortável
    aplicar(max(mu['muito_alta'], mu['muito_baixa']), conforto_nao)
    # 8: vento forte -> não confortável
    aplicar(mv['forte'], conforto_nao)

    # Defuzzificação
    if not agregada.any():
        score = 0.0
    else:
        score = fuzz.defuzz(conforto, agregada, 'centroid')

    if score >= 66:
        categoria = 'Confortável'
    elif score >= 33:
        categoria = 'Marginal'
    else:
        categoria = 'Não confortável'

    return score, categoria

def classificar_temperatura(temperatura_atual):
    muito_frio   = fuzz.interp_membership(temperatura, grau_muito_frio,   temperatura_atual)
    frio         = fuzz.interp_membership(temperatura, grau_frio,         temperatura_atual)
    medio        = fuzz.interp_membership(temperatura, grau_medio,        temperatura_atual)
    quente       = fuzz.interp_membership(temperatura, grau_quente,       temperatura_atual)
    muito_quente = fuzz.interp_membership(temperatura, grau_muito_quente, temperatura_atual)

    graus = {
        "muito fria":  muito_frio,
        "fria":        frio,
        "média":       medio,
        "quente":      quente,
        "muito quente": muito_quente
    }

    classe = max(graus, key=graus.get)  # pega a maior pertinência
    return classe

def classificar_umidade(umidade_atual):
    muito_baixa = fuzz.interp_membership(umidade, umidade_muito_baixa, umidade_atual)
    baixa       = fuzz.interp_membership(umidade, umidade_baixa,       umidade_atual)
    moderada    = fuzz.interp_membership(umidade, umidade_moderada,    umidade_atual)
    alta        = fuzz.interp_membership(umidade, umidade_alta,        umidade_atual)
    muito_alta  = fuzz.interp_membership(umidade, umidade_muito_alta,  umidade_atual)

    graus = {
        "muito baixa":  muito_baixa,
        "baixa":        baixa,
        "moderada":     moderada,
        "alta":         alta,
        "muito alta":   muito_alta
    }

    classe = max(graus, key=graus.get)
    return classe

def classificar_vento(vento_atual):
    fraco    = fuzz.interp_membership(vento, vento_fraco,    vento_atual)
    moderado = fuzz.interp_membership(vento, vento_moderado, vento_atual)
    forte    = fuzz.interp_membership(vento, vento_forte,    vento_atual)

    graus = {
        "fraco":    fraco,
        "moderado": moderado,
        "forte":    forte
    }

    classe = max(graus, key=graus.get)
    return classe

def sugerir_atividade_com_excel(temperatura_classificacao, umidade_classificacao, vento_classificacao):
    recomendacoes = ler_recomendacoes()

    # filtra as recomendações para buscar no excel
    recomendacao = recomendacoes[
        (recomendacoes['Temperatura'] == temperatura_classificacao) &
        (recomendacoes['Umidade'] == umidade_classificacao) &
        (recomendacoes['Vento'] == vento_classificacao)
    ]
    
    if not recomendacao.empty:
        return recomendacao['Atividade Sugerida'].values[0]
    else:
        return "Não há recomendação disponível para essas condições."

def mostrar_recomendacao():
    cidade = cidade_entry.get()
    
    lat, lon = obter_coordenadas(cidade)
    
    if lat is not None and lon is not None:
        dados_climaticos = obter_dados_climaticos(cidade, lat, lon)
        
        if dados_climaticos:
            temperatura_atual = dados_climaticos["temperature"]
            umidade_atual = dados_climaticos["humidity"]
            vento_atual = dados_climaticos["wind_speed"]
            
            temperatura_classificacao = classificar_temperatura(temperatura_atual)
            umidade_classificacao = classificar_umidade(umidade_atual)
            vento_classificacao = classificar_vento(vento_atual)

            # Avalia conforto para exercício
            score_conforto, categoria_conforto = avaliar_conforto(temperatura_atual, umidade_atual, vento_atual)

            sugestao_exercicio = (
                "Aconselhável fazer exercício." if categoria_conforto == 'Confortável'
                else "Não aconselhável fazer exercício no momento."
            )

            resultado = (
                f"Temperatura: {temperatura_atual}°C - {temperatura_classificacao}\n"
                f"Umidade: {umidade_atual}% - {umidade_classificacao}\n"
                f"Vento: {vento_atual} km/h - {vento_classificacao}\n\n"
                f"Conforto para exercício: {score_conforto:.1f}/100 - {categoria_conforto}\n"
                f"{sugestao_exercicio}\n\n"
                f"{sugerir_atividade_com_excel(temperatura_classificacao, umidade_classificacao, vento_classificacao)}"
            )
            
            resultado_label.config(text=resultado)
        else:
            messagebox.showerror("Erro", "Não foi possível obter os dados climáticos.")
    else:
        messagebox.showerror("Erro", "Cidade não encontrada ou coordenadas não obtidas.")

root = tk.Tk()
root.title("Sistema de Classificação de Sensação Térmica")
root.config(bg="#8fd2ef")  
titulo_label = tk.Label(root, text="Classificação de Sensação Térmica", font=("Helvetica", 18, "bold"), bg="#8fd2ef", fg="black")
titulo_label.pack(pady=20)
cidade_label = tk.Label(root, text="Digite o nome da cidade:", font=("Helvetica", 14), bg="#8fd2ef", fg="black")
cidade_label.pack()
cidade_entry = tk.Entry(root, font=("Roboto", 14), width=20, bd=3, relief="flat", fg="black")
cidade_entry.pack(pady=10)
botao = tk.Button(root, text="Buscar dados", font=("Roboto", 14), bg="#d7eaf2", fg="black", bd=0, relief="raised", command=mostrar_recomendacao)
botao.pack(pady=20)
resultado_label = tk.Label(root, text="", font=("Roboto", 14), bd=2, bg="#8fd2ef", fg="black", justify="center", anchor="center")
resultado_label.pack(pady=10)
root.mainloop()