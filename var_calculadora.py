# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import io

# ========= Configuração da página ==========
st.set_page_config(page_title="Calculadora de VaR", layout="wide")
st.title("📊 Calculadora Interativa de Value at Risk (VaR)")
st.markdown(
    """
    Esta aplicação calcula o **Value at Risk (VaR)** de uma carteira de ações usando três métodos:
    - 📈 **Histórico**
    - 📊 **Paramétrico (Normal)**
    - 🎲 **Monte Carlo**
    
    Inclui também teste de **backtesting (Kupiec)** para verificar a adequação do modelo.
    """
)

# ========= Funções auxiliares ==========

def get_data_and_returns(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(how="all").dropna(axis=1)
    valid_tickers = list(prices.columns)
    if not valid_tickers:
        st.error("❌ Nenhum ticker foi baixado com sucesso.")
        return None
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets, valid_tickers

def var_historico(returns, alpha):
    q = returns.quantile(1 - alpha)
    return max(0.0, -q)

def var_parametrico(mu, sigma, alpha):
    z = norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    return float(max(0.0, var))

def var_mc(mu_vec, cov, w, alpha, n_sims):
    sims = np.random.multivariate_normal(mean=mu_vec, cov=cov, size=n_sims)
    port_sims = sims @ w
    q = np.quantile(port_sims, 1 - alpha)
    return float(max(0.0, -q))

def kupiec_test(returns, var_values, alpha):
    var_values_pct = var_values / 100.0
    violations = (returns < -var_values_pct).astype(int)
    T = len(violations)
    N = violations.sum()
    p = 1 - alpha
    if T < 30:
        return N, np.nan
    p_hat = N / T
    if N in [0, T]:
        return N, np.nan
    LR_uc = -2 * (
        np.log(((1 - p)**(T - N)) * (p**N)) - np.log(((1 - p_hat)**(T - N)) * (p_hat**N))
    )
    p_value = 1 - chi2.cdf(LR_uc, 1)
    return N, p_value

# ========= Painel lateral =========
with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    tickers = st.text_input(
        "Tickers (separados por vírgula)", "VBBR3.SA, MCD, UBER, VALE3.SA, GS"
    )
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    start = st.date_input("Data inicial", pd.to_datetime("2022-01-01"))
    end = st.date_input("Data final", pd.Timestamp.today())
    confidence_levels = st.multiselect(
        "Níveis de confiança", [0.95, 0.975, 0.99], default=[0.95, 0.99]
    )
    horizon = st.number_input("Horizonte (dias)", min_value=1, value=1)
    metodo = st.selectbox(
        "Método de VaR", ["Histórico", "Paramétrico (Normal)", "Monte Carlo"]
    )
    n_sims = st.number_input("Nº de Simulações (Monte Carlo)", 10_000, 200_000, 100_000, step=10_000)
    usar_pesos = st.checkbox("Inserir pesos personalizados")
    run = st.button("🚀 Calcular VaR")

# ========= Execução =========
if run:
    with st.spinner("Baixando dados e calculando..."):
        result = get_data_and_returns(tickers, start, end)

        if result:
            rets, valid_tickers = result
            mu_vec = rets.mean().values
            sigma_vec = rets.std().values
            cov_mat = rets.cov().values

            if usar_pesos:
                st.subheader("⚖️ Defina os pesos dos ativos")
                w_inputs = []
                for t in valid_tickers:
                    w = st.number_input(f"Peso de {t}", min_value=0.0, max_value=1.0, value=1/len(valid_tickers))
                    w_inputs.append(w)
                w = np.array(w_inputs)
                w = w / w.sum()
            else:
                w = np.array([1 / len(valid_tickers)] * len(valid_tickers))

            port_ret = rets @ w
            mu_p, sigma_p = port_ret.mean(), port_ret.std()

            st.success(f"✅ Tickers válidos: {valid_tickers}")
            st.write("Pesos da carteira:", np.round(w, 3))

            # ======= Cálculo do VaR =======
            resultados = []
            for alpha in confidence_levels:
                if metodo == "Histórico":
                    var_val = var_historico(port_ret, alpha)
                elif metodo == "Paramétrico (Normal)":
                    var_val = var_parametrico(mu_p, sigma_p, alpha)
                else:
                    var_val = var_mc(mu_vec, cov_mat, w, alpha, n_sims)
                resultados.append({"Confiança": alpha, "VaR_%": 100 * var_val})

            df_var = pd.DataFrame(resultados)

            st.subheader(f"📉 Resultado do VaR — Método: {metodo}")
            st.dataframe(df_var.style.format(precision=3))

            # ======= Gráfico =======
            fig, ax = plt.subplots()
            ax.bar(df_var["Confiança"].astype(str), df_var["VaR_%"])
            ax.set_ylabel("VaR (% do patrimônio)")
            ax.set_xlabel("Nível de Confiança")
            ax.set_title(f"VaR da Carteira ({metodo})")
            st.pyplot(fig)

            # ======= Backtesting =======
            st.subheader("🧪 Backtesting (Kupiec Test)")
            window = 252
            alpha = confidence_levels[0]
            var_hist_movel = -port_ret.rolling(window=window).quantile(1 - alpha).dropna() * 100
            rets_backtest = port_ret.loc[var_hist_movel.index]
            N, p_value = kupiec_test(rets_backtest, var_hist_movel, alpha)

            st.write(f"Nível de confiança: {alpha:.1%}")
            st.write(f"Violações observadas: {N}")
            st.write(f"P-valor do teste: {p_value:.4f}")
            st.write("Adequação do modelo:", "✅ Sim" if p_value > 0.05 else "❌ Não")

            # ======= Download Excel =======
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_var.to_excel(writer, sheet_name="VaR_Carteira", index=False)
            st.download_button(
                label="💾 Baixar resultados (Excel)",
                data=buffer.getvalue(),
                file_name="resultados_var.xlsx",
                mime="application/vnd.ms-excel",
            )
