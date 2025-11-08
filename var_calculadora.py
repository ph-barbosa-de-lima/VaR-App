# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Calculadora de VaR", layout="wide")

# Funções utilitárias (adaptadas do seu script)
def get_data_and_returns(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(how="all").dropna(axis=1)
    valid_tickers = list(prices.columns)

    if not valid_tickers:
        st.error("❌ Nenhum ticker foi baixado com sucesso.")
        return None

    weights = np.array([1/len(valid_tickers)] * len(valid_tickers))
    rets = np.log(prices / prices.shift(1)).dropna()

    mu_vec = rets.mean().values
    cov_mat = rets.cov().values
    sigma_vec = rets.std().values
    port_ret = rets @ weights
    mu_p = port_ret.mean()
    sigma_p = port_ret.std()

    return rets, valid_tickers, weights, mu_vec, cov_mat, sigma_vec, port_ret, mu_p, sigma_p

def var_historico(returns, alpha):
    q = returns.quantile(1 - alpha)
    return max(0.0, -q)

def var_parametrico_normal(mu, sigma, alpha):
    z = norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    return float(max(0.0, var))

def var_mc_normal_multivariado(mu_vec, cov, w, alpha, n_sims=100_000):
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
    LR_uc = -2 * (np.log(((1 - p)**(T - N)) * (p**N)) - np.log(((1 - p_hat)**(T - N)) * (p_hat**N)))
    p_value = 1 - chi2.cdf(LR_uc, 1)
    return N, p_value

# ===========================
# Interface Streamlit
# ===========================
st.title("📊 Calculadora de Value at Risk (VaR)")
st.markdown("Ferramenta interativa para cálculo e teste de VaR de uma carteira de ações.")

with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    tickers = st.text_input("Tickers (separados por vírgula)", "VBBR3.SA, MCD, UBER, VALE3.SA, GS")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    start = st.date_input("Data inicial", pd.to_datetime("2022-01-01"))
    end = st.date_input("Data final", pd.Timestamp.today())
    confidence_levels = st.multiselect("Níveis de confiança", [0.95, 0.975, 0.99], default=[0.95, 0.99])
    horizon = st.number_input("Horizonte (dias)", min_value=1, value=1)
    run = st.button("🔍 Calcular VaR")

if run:
    with st.spinner("Baixando dados e calculando..."):
        result = get_data_and_returns(tickers, start, end)

        if result:
            rets, valid_tickers, w, mu_vec, cov_mat, sigma_vec, port_ret, mu_p, sigma_p = result

            st.success(f"✅ Tickers válidos: {valid_tickers}")
            st.write("Pesos da carteira:", np.round(w, 3))

            # Resultados de VaR
            df_port = []
            for alpha in confidence_levels:
                vh = var_historico(port_ret, alpha)
                vp = var_parametrico_normal(mu_p, sigma_p, alpha)
                vmc = var_mc_normal_multivariado(mu_vec, cov_mat, w, alpha)
                df_port.append({
                    "Confiança": alpha,
                    "VaR_Histórico_%": 100 * vh,
                    "VaR_Param_Norm_%": 100 * vp,
                    "VaR_MonteCarlo_%": 100 * vmc
                })
            df_port = pd.DataFrame(df_port)

            st.subheader("📉 VaR da Carteira (% do patrimônio)")
            st.dataframe(df_port.style.format(precision=3))

            # Gráfico
            fig, ax = plt.subplots()
            df_port.set_index("Confiança")[["VaR_Histórico_%","VaR_Param_Norm_%","VaR_MonteCarlo_%"]].plot(kind="bar", ax=ax)
            ax.set_ylabel("VaR (%)")
            st.pyplot(fig)

            # Backtesting simples (VaR histórico)
            window = 252
            alpha = confidence_levels[0]
            var_hist_movel = -port_ret.rolling(window=window).quantile(1 - alpha).dropna() * 100
            rets_backtest = port_ret.loc[var_hist_movel.index]
            N, p_value = kupiec_test(rets_backtest, var_hist_movel, alpha)

            st.subheader("🧪 Backtesting (Teste de Kupiec)")
            st.write(f"Confiança: {alpha:.1%}")
            st.write(f"Violações observadas: {N}")
            st.write(f"P-valor do teste: {p_value:.4f}")
            st.write("Adequação do modelo:", "✅ Sim" if p_value > 0.05 else "❌ Não")

            # Download em Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_port.to_excel(writer, sheet_name="VaR_Carteira", index=False)
            st.download_button(
                label="💾 Baixar resultados (Excel)",
                data=buffer.getvalue(),
                file_name="resultados_var.xlsx",
                mime="application/vnd.ms-excel",
            )
