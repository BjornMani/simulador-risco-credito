import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="Simulador de Risco de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Funções de Carga ---
@st.cache_resource
def load_model_assets(segmento):
    """
    Carrega o Modelo, Scaler, Lista de Colunas e Metadados para o segmento escolhido.
    """
    base_path = Path("models")
    
    try:
        model = joblib.load(base_path / f"model_{segmento}.pkl")
        scaler = joblib.load(base_path / f"scaler_{segmento}.pkl")
        cols_df = pd.read_csv(base_path / f"columns_{segmento}.csv")
        cols = cols_df.columns.tolist()
        meta = joblib.load(base_path / f"meta_{segmento}.pkl")
        
        return model, scaler, cols, meta
    except FileNotFoundError:
        return None, None, None, None

# --- 3. Interface Principal ---

# Cabeçalho
st.title("🏦 Cockpit de Risco de Crédito")
st.markdown("---")

col_info, col_select = st.columns([2, 1])

with col_info:
    st.markdown("""
    Este simulador utiliza modelos de **Machine Learning (Random Forest)** treinados para prever a tendência da inadimplência.
    
    **Como funciona?**
    O modelo prevê a **variação (delta)** esperada com base em choques macroeconômicos.
    """)

with col_select:
    segmento_escolhido = st.selectbox(
        "Selecione a Carteira:",
        ["PF", "PJ", "Rural_PF", "Rural_PJ"],
        index=0
    )

# Carga dos Dados
model, scaler, feature_cols, meta = load_model_assets(segmento_escolhido)

if model is None:
    st.error(f"❌ Modelo para '{segmento_escolhido}' não encontrado na pasta 'models/'. Execute o notebook 05 primeiro.")
    st.stop()

# --- 4. Sidebar: Controle de Cenários ---
st.sidebar.header(f"⚙️ Cenário: {segmento_escolhido}")
st.sidebar.markdown("Defina os indicadores macroeconômicos para o próximo mês.")

# X_ultimo_real é um dicionário salvo no passo 05
last_features = meta.get("X_ultimo_real", {})

input_values = {}

# 1. Selic (Variável chave)
default_selic = last_features.get('selic', 10.75)
selic_input = st.sidebar.slider("Selic (%)", 2.0, 20.0, float(default_selic), 0.25)
input_values['selic'] = selic_input

# 2. IPCA (Inflação)
default_ipca = last_features.get('ipca', 0.5)
ipca_input = st.sidebar.slider("IPCA Mensal (%)", -1.0, 2.0, float(default_ipca), 0.1)
input_values['ipca'] = ipca_input

# 3. Dólar (Importante para Rural/PJ)
default_dolar = last_features.get('dolar_ptax', 5.0)
dolar_input = st.sidebar.number_input("Dólar (R$)", 3.0, 8.0, float(default_dolar), 0.1)
input_values['dolar_ptax'] = dolar_input

# 4. Inadimplência Anterior (Inércia)
valor_referencia_real = meta['valor_ultimo_real']
inad_anterior_simulada = st.sidebar.number_input(
    f"Inadimplência Atual (%) - Ref: {meta['data_referencia'].date()}", 
    0.0, 20.0, float(valor_referencia_real), 0.1,
    help="O modelo usa o mês anterior como ponto de partida (Inércia)."
)
input_values['inad_anterior'] = inad_anterior_simulada

st.sidebar.markdown("---")
st.sidebar.caption("*Demais variáveis são mantidas no último valor observado (Ceteris Paribus).*")

# --- 5. Processamento da Previsão ---

# A. Montar o DataFrame de Entrada
df_input = pd.DataFrame([last_features])

# Garantir que o DF tenha todas as colunas esperadas pelo modelo, na ordem certa
for col in feature_cols:
    if col not in df_input.columns:
        df_input[col] = 0

# Reordenar colunas
df_input = df_input[feature_cols]

# B. Atualizar com os inputs do usuário
for col, val in input_values.items():
    if col in df_input.columns:
        df_input[col] = val
    
    if f"{col}_lag_3" in df_input.columns: df_input[f"{col}_lag_3"] = val
    if f"{col}_lag_6" in df_input.columns: df_input[f"{col}_lag_6"] = val

# C. Escalar e Prever
X_scaled = scaler.transform(df_input)
delta_pred = model.predict(X_scaled)[0]

# D. Calcular Resultado Final
# Previsão = Ponto de Partida (Simulado) + Variação Prevista
previsao_final = inad_anterior_simulada + delta_pred

# Trava de segurança (não existe inadimplência negativa)
previsao_final = max(0.0, previsao_final)

# --- 6. Exibição dos Resultados (Dashboard) ---

st.subheader(f"📊 Resultados da Simulação: {segmento_escolhido}")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Cenário Atual (Base)", 
        f"{inad_anterior_simulada:.2f}%",
        help="Ponto de partida informado na sidebar"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Variação Prevista (Delta)", 
        f"{delta_pred:+.2f} p.p.",
        delta_color="inverse" 
    )
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card" style="border: 2px solid #4CAF50;">', unsafe_allow_html=True)
    st.metric(
        "Previsão Próximo Mês", 
        f"{previsao_final:.2f}%",
        delta=f"{delta_pred:+.2f}",
        delta_color="inverse"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. Gráfico de Sensibilidade ---
st.markdown("---")
st.subheader("🔎 Análise de Sensibilidade: Selic vs Inadimplência")
st.markdown("Como a taxa de juros impacta este modelo específico, mantendo os outros fatores constantes?")

# Gerar dados para o gráfico
selic_range = np.linspace(2.0, 25.0, 40)
preds_sensibilidade = []

# Loop para simular vários cenários de Selic
X_temp = df_input.copy()
idx_selic = df_input.columns.get_loc("selic") if "selic" in df_input.columns else -1

if idx_selic != -1:
    # Matriz repetida (otimização numpy)
    X_sens = np.tile(X_temp.values, (len(selic_range), 1))
    # Substituir coluna Selic
    X_sens[:, idx_selic] = selic_range
    # Escalar
    X_sens_scaled = scaler.transform(X_sens)
    # Prever Deltas
    deltas = model.predict(X_sens_scaled)
    # Somar à base
    preds_final = inad_anterior_simulada + deltas
    
    # Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selic_range, 
        y=preds_final,
        mode='lines',
        name='Curva de Reação',
        line=dict(color='#ff4b4b', width=3)
    ))
    
    # Marcador do ponto escolhido
    fig.add_trace(go.Scatter(
        x=[selic_input],
        y=[previsao_final],
        mode='markers',
        name='Sua Escolha',
        marker=dict(color='black', size=12)
    ))
    
    fig.update_layout(
        title="Curva de Sensibilidade da Selic",
        xaxis_title="Taxa Selic (%)",
        yaxis_title=f"Inadimplência Prevista {segmento_escolhido} (%)",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("A variável 'selic' não foi encontrada nas features deste modelo específico.")

# Rodapé
st.caption("Desenvolvido para análise estratégica de risco. Modelo preditivo v1.0")