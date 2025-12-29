#  Relatório Técnico: Modelagem de Risco de Crédito (Inadimplência PF/PJ/Rural)

**Autor:** Pedro Rogério Pereira Júnior

**Data:** *Dezembro/2025*

**Link do Simulador:** *``https://simulador-risco-credito-inadimplencia.streamlit.app/``*

## Arquitetura: 
projeto_inadimplencia/ 
├── data/  
│   ├── processed/  
│   │   ├── df_modelagem_v3.csv  
│   │   ├── df_modelagem.csv  
│   │   └── X_train_sample.csv  
│   │
│   └── raw/  
│       ├── df_economico.csv  
│       ├── df_eventos_politicos.csv  
│       ├── df_ibge.csv  
│       ├── df_inadimplencia.csv  
│       ├── df_inmet.csv  
│       └── INMET.zip   
│
├── models/  
│   ├── columns_PF.csv  
│   ├── columns_PJ.csv  
│   ├── columns_Rural_PF.csv  
│   ├── columns_Rural_PJ.csv  
│   ├── last_values_PF.csv  
│   ├── last_values_PJ.csv
│   ├── last_values_Rural_PF.csv  
│   ├── last_values_Rural_PJ.csv  
│   ├── model_PF.pkl  
│   ├── model_PJ.pkl  
│   ├── model_Rural_PF.pkl  
│   ├── model_Rural_PJ.pkl  
│   ├── scaler_PF.pkl  
│   ├── scaler_PJ.pkl  
│   ├── scaler_Rural_PF.pkl    
│   └── scaler_Rural_PJ.pkl 
│
├── notebooks/  
│   ├── 01_coleta_variaveis.ipynb  
│   ├── 02_analise_exploratoria.ipynb  
│   ├── 03_feature_engineering.ipynb  
│   ├── 04_treinamento_modelo.ipynb  
│   ├── 05_treinamento_multimodelo.ipynb  
│   ├── 06_treinamento_comparativo.ipynb  
│   ├── 07_treinamento_comparativo_delta.ipynb  
│   └── 08_treinamento_focado.ipynb   
│
├── reports/  
├── src/  
├── venv/  
├── app_1.py  
├── app_2.py 
├── app.py  
├── readme.md    
└── requirements.txt


## 1. O Problema de Negócio
O objetivo deste projeto foi desenvolver um motor preditivo capaz de antecipar o comportamento da Inadimplência (Pessoa Física, Jurídica e Rural) frente a choques macroeconômicos (Selic, Inflação e Câmbio) em um horizonte de 18 meses.

A principal dor resolvida é a **falta de sensibilidade** dos modelos tradicionais de curto prazo, que tendem a replicar o passado recente (inércia) e falham em simular cenários de stress (ex: disparada do Dólar ou da Selic).

## 2. Metodologia e Evolução dos Modelos
O desenvolvimento passou por **três fases** distintas até atingir a solução final.
- **Fase 1:** *Random Forest & XGBoost* (Abordagem Tradicional)
    - Inicialmente, foram treinados modelos de árvore (*Ensemble*) utilizando mais de 50 variáveis macroeconômicas e *lags* (atrasos) temporais.
    - **Resultado:** Alta precisão no curto prazo (RMSE baixo), mas **baixa sensibilidade** a cenários.
    - **Problema Detectado:** Os modelos "ancoravam" na variável ``inadimplencia_lag_1`` (valor do mês anterior) e ignoravam as mudanças na Selic simulada, resultando em linhas retas nas projeções de longo prazo.
- **Fase 2:** *Ridge Regression* (Abordagem Econométrica)
    - Para corrigir a insensibilidade, migramos para modelos lineares regularizados (Ridge).
    - **Resultado:** O modelo passou a respeitar a correlação (Se Selic sobe, Inadimplência sobe), mas sofria com **"Explosão de Coeficientes"**, gerando previsões irreais (ex: 180% de inadimplência) devido à multicolinearidade entre variáveis redundantes (Selic vs CDI vs Spread).
- **Fase 3:** Modelo de Sensibilidade Focada (Solução Final)
    - Adotamos uma abordagem de *Feature Selection* Cirúrgica. O modelo final foi treinado utilizando apenas as **"Alavancas de Controle"**:
    - **Selic** (Defasada em 6 meses): Principal driver para PF e PJ Urbana.
    - **IPCA**: Proxy para corrosão de renda.
    - **Dólar PTAX**: Principal driver para o Crédito Rural (receita de exportação).
    - **Sazonalidade** (Safra/Mês): Para capturar ciclos do Agro e Varejo.
    - Essa abordagem sacrificou marginalmente a precisão do "próximo mês" em troca de uma **alta capacidade de simulação de tendências (*Stress Testing*)**, que era o objetivo principal do simulador.

## 3. Arquitetura da Solução
- **Linguagem:** *Python* 3.10
- **Coleta de Dados:** Séries temporais do Banco Central (SGS), IBGE e INMET.
- **Engine de ML:** *Scikit-Learn* (*Ridge Regression* e *StandardScaler*).
- **Frontend:** *Streamlit* (Hospedado em *Streamlit Cloud*).
- **Reprodutibilidade:** Controle de dependências via ``requirements.txt`` e versionamento via Git.

## 4. Principais *Insights* (Resultados)
A modelagem revelou dinâmicas distintas entre os setores:

|**Carteira**|***Driver* Principal**|**Comportamento Observado**|
|---|---|---|
|**Pessoa Física**|Juros (Selic)|Alta elasticidade. Aumento de juros impacta a inadimplência com lag de 6 a 9 meses.|
|**Pessoa Jurídica**|Juros + Atividade|Segue padrão similar à PF, mas com maior volatilidade.|
|**Rural (Agro)**|Câmbio (Dólar)|Correlação inversa. Dólar Alto reduz a inadimplência (aumenta receita do exportador). Dólar Baixo é o principal fator de risco para esta carteira.|

## 5. Como Utilizar o Simulador
- 1º - Acesse o painel online.
- 2º - Atualize os valores iniciais (Selic/Dólar) com os dados de mercado do dia (visto que o modelo parte do último dado histórico do dataset).
- 3º - Utilize os sliders de tendência para simular choques:
	- Ex: O que acontece com a carteira Rural se o Dólar cair R$ 0,20 ao mês pelos próximos 18 meses?
- 4º - O gráfico projetará a curva de inadimplência esperada para o cenário definido.