import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="Arena de Modelos - Risco de Crédito",
    page_icon="⚔️",
    layout="wide"
)

# Estilização Customizada
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

st.title("⚔️ Arena de Modelos: Simulador de Cenários")
st.markdown("""
Compare como diferentes "cérebros" de Inteligência Artificial reagem a choques econômicos.
**Objetivo:** Entender a sensibilidade da carteira a mudanças na taxa de juros (Selic) nos próximos 18 meses.
""")

# --- 2. Carga de Artefatos ---
@st.cache_resource
# Trecho ajustado da função load_assets no app_2.py

def load_assets(segmento, algoritmo_nome):
    base_path = Path("models")
    
    try:
        model = joblib.load(base_path / f"model_{segmento}_{algoritmo_nome}.pkl")
        
        # --- PROTEÇÃO CONTRA O ERRO DE 50 vs 45 ---
        # Verifica quantas features o modelo espera
        n_features_model = getattr(model, "n_features_in_", None)
        
        # Carrega o Scaler
        scaler = joblib.load(base_path / f"scaler_{segmento}.pkl")
        
        # Verifica se o Scaler bate com o Modelo
        if n_features_model and scaler.n_features_in_ != n_features_model:
            st.error(f"ERRO CRÍTICO: O Scaler tem {scaler.n_features_in_} colunas, mas o Modelo quer {n_features_model}.")
            st.warning("Solução: Delete os arquivos da pasta 'models/' e rode o Notebook de treinamento novamente.")
            return None, None, None, None

        cols = pd.read_csv(base_path / f"columns_{segmento}.csv").columns.tolist()
        last_vals = pd.read_csv(base_path / f"last_values_{segmento}.csv", index_col=0).squeeze()
        
        return model, scaler, cols, last_vals
        
    except FileNotFoundError:
        return None, None, None, None

# --- 3. Motor de Simulação Dinâmica ---
def run_simulation(model, scaler, feature_names, initial_input, start_inad, selic_trend, months=18):
    """
    Simula o futuro mês a mês.
    Lógica:
    1. Atualiza Selic baseada na tendência escolhida.
    2. Atualiza Sazonalidade (Mês).
    3. Prever o Delta.
    4. Atualiza o valor acumulado.
    """
    predictions = []
    current_inad = start_inad
    
    # Prepara o dicionário de input inicial
    current_input = initial_input.copy()
    
    # Descobre qual mês estamos (pela feature 'mes' ou data atual)
    current_month = int(current_input.get('mes', datetime.now().month))
    
    # Valor base da Selic (pega o lag mais recente disponível ou define 10.5 como padrão)
    selic_base = current_input.get('selic_lag_6', current_input.get('selic', 10.5))
    
    for i in range(months):
        # --- A. Choque na Economia (Cenário) ---
        delta_selic = selic_trend * (i + 1)
        new_selic = max(2.0, min(30.0, selic_base + delta_selic))
        
        for col in feature_names:
            if 'selic' in col.lower():
                current_input[col] = new_selic
        
        # --- B. Atualiza Sazonalidade ---
        current_month += 1
        if current_month > 12: current_month = 1
        
        if 'mes' in feature_names: current_input['mes'] = current_month
        
        # --- C. Previsão ---
        df_step = pd.DataFrame([current_input])
        
        # Garante ordem e colunas corretas (preenche faltantes com 0)
        df_step = df_step.reindex(columns=feature_names, fill_value=0)
        
        # Escala e Prevê
        X_scaled = scaler.transform(df_step)
        delta_pred = model.predict(X_scaled)[0]
        
        # --- D. Acumulação ---
        current_inad += delta_pred
        
        # Trava de segurança (não existe inadimplência < 0)
        current_inad = max(0.0, current_inad)
        predictions.append(current_inad)
        
    return predictions

# --- 4. Sidebar: Configuração da IA ---
st.sidebar.header("🧠 Configuração da IA")

algo_options = {
    "RandomForest": "RandomForest",
    "XGBoost": "XGBoost",
    "Ridge (Linear)": "Ridge"
}

nome_amigavel = st.sidebar.selectbox("Escolha o Algoritmo:", list(algo_options.keys()))
algoritmo_chave = algo_options[nome_amigavel]

st.sidebar.info(f"""
**{nome_amigavel}**:
{'Detecta padrões complexos e não-lineares.' if 'Forest' in nome_amigavel else ''}
{'Focado em corrigir erros e alta performance.' if 'XGB' in nome_amigavel else ''}
{'Simples, robusto e mostra a tendência macro.' if 'Ridge' in nome_amigavel else ''}
""")

# --- 5. Interface Principal (Tabs) ---

# Mapeamento: Nome na Aba -> Sufixo do Arquivo
segmentos = {
    "👤 Pessoa Física": "PF",
    "🏢 Pessoa Jurídica": "PJ",
    "🚜 Rural PF": "Rural_PF",
    "🚜 Rural PJ": "Rural_PJ"
}

tabs = st.tabs(list(segmentos.keys()))

for aba_nome, segmento_id in segmentos.items():
    with tabs[list(segmentos.keys()).index(aba_nome)]:
        
        # Carregar Modelo Específico
        model, scaler, cols, last_vals = load_assets(segmento_id, algoritmo_chave)
        
        if model is None:
            st.warning(f"⚠️ Modelo '{algoritmo_chave}' para '{segmento_id}' não encontrado.")
            st.caption("Dica: Verifique se rodou o notebook '06_treinamento_comparativo.ipynb' ou '07'.")
            continue

        # --- Layout de Colunas ---
        c_settings, c_chart = st.columns([1, 3])
        
        with c_settings:
            st.subheader("Parâmetros")
            
            # Ponto de Partida
            val_inicial = float(last_vals.get('target_lag_1', 3.0))
            start_inad = st.number_input(
                "Inadimplência Atual (%)", 
                value=val_inicial, format="%.2f", step=0.1, 
                key=f"start_{segmento_id}"
            )
            
            st.markdown("---")
            st.write("**Cenário Selic**")
            
            # Slider de Tendência
            selic_trend = st.slider(
                "Variação Mensal (pontos base)",
                min_value=-0.5, max_value=0.5, value=0.0, step=0.05,
                format="%+.2f pp",
                key=f"trend_{segmento_id}",
                help="Ex: +0.10 significa que a Selic sobe 0.10% todo mês."
            )
            
            total_change = selic_trend * 12
            st.caption(f"Impacto em 1 ano: **{total_change:+.2f}% na Selic**")

        with c_chart:
            # --- Executar Simulação ---
            try:
                # 1. Simulação "Base" (Selic Constante)
                pred_base = run_simulation(model, scaler, cols, last_vals, start_inad, selic_trend=0.0)
                
                # 2. Simulação "Cenário" (Com a tendência escolhida)
                pred_scenario = run_simulation(model, scaler, cols, last_vals, start_inad, selic_trend=selic_trend)
                
                # --- Plotagem com Plotly ---
                fig = go.Figure()
                
                meses = list(range(1, 19))
                
                # Linha Base (Cinza)
                fig.add_trace(go.Scatter(
                    x=meses, y=pred_base,
                    mode='lines',
                    name='Cenário Estável',
                    line=dict(color='gray', width=2, dash='dot'),
                    opacity=0.6
                ))
                
                # Linha Cenário (Colorida)
                cor_linha = '#ff4b4b' if pred_scenario[-1] > start_inad else '#00C853'
                fig.add_trace(go.Scatter(
                    x=meses, y=pred_scenario,
                    mode='lines+markers',
                    name=f'Cenário Simulad ({algoritmo_chave})',
                    line=dict(color=cor_linha, width=4)
                ))
                
                # Layout
                fig.update_layout(
                    title=f"Projeção de 18 Meses: {aba_nome}",
                    xaxis_title="Meses à Frente",
                    yaxis_title="Taxa de Inadimplência (%)",
                    hovermode="x unified",
                    height=500,
                    template="plotly_white",
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Métricas Finais
                delta_total = pred_scenario[-1] - start_inad
                st.metric(
                    label="Projeção para o Mês 18",
                    value=f"{pred_scenario[-1]:.2f}%",
                    delta=f"{delta_total:+.2f} p.p. acumulados",
                    delta_color="inverse" # Vermelho se subir
                )
                
            except Exception as e:
                st.error("Erro na Simulação.")
                st.exception(e)

# Rodapé
st.markdown("---")
st.caption("Sistema de Inteligência Competitiva de Crédito • v2.0 Pro")