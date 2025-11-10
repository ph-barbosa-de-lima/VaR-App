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

st.markdown("""
Ferramenta interativa para cálculo e visualização do **Value at Risk (VaR)**:
- 📈 Histórico
- 📊 Paramétrico (Normal)
- 🎲 Monte Carlo
- 📉 Curva temporal de VaR e Backtesting
""")

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
    tickers = st.text_input("Tickers (separados por vírgula)", "VBBR3.SA, MCD, UBER, VALE3.SA, GS")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    start = st.date_input("Data inicial", pd.to_datetime("2020-01-01"))
    end = st.date_input("Data final", pd.Timestamp.today())
    confidence = st.selectbox("Nível de confiança", [0.95, 0.975, 0.99], index=0)
    metodo = st.selectbox("Método de VaR", ["Histórico", "Paramétrico (Normal)", "Monte Carlo"])
    n_sims = st.number_input("Simulações (Monte Carlo)", 10_000, 200_000, 100_000, step=10_000)
    usar_pesos = st.checkbox("Inserir pesos personalizados")
    run = st.button("🚀 Calcular e Mostrar Curva de VaR")

# ========= Execução =========
if run:
    result = get_data_and_returns(tickers, start, end)
    if result:
        rets, valid_tickers = result
        mu_vec, sigma_vec, cov_mat = rets.mean().values, rets.std().values, rets.cov().values

        # Pesos
        if usar_pesos:
            st.markdown("### ⚖️ Pesos personalizados")
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

        # ========== VaR Estático ==========
        if metodo == "Histórico":
            var_val = var_historico(port_ret, confidence)
        elif metodo == "Paramétrico (Normal)":
            var_val = var_parametrico(mu_p, sigma_p, confidence)
        else:
            var_val = var_mc(mu_vec, cov_mat, w, confidence, n_sims)

        st.subheader("📉 VaR Estático da Carteira")
        st.metric(f"VaR {int(confidence*100)}%", f"{100 * var_val:.3f}%")

        # ========== Curva temporal do VaR ==========
        window = 252
        st.subheader("📊 Curva Temporal do VaR e Retornos")

        if metodo == "Histórico":
            var_curve = -port_ret.rolling(window).quantile(1 - confidence)
        else:
            mu_roll = port_ret.rolling(window).mean()
            sigma_roll = port_ret.rolling(window).std()
            z = norm.ppf(1 - confidence)
            var_curve = -(mu_roll + z * sigma_roll)

        var_curve = var_curve.dropna() * 100
        port_ret_pct = port_ret.loc[var_curve.index] * 100

        # Violations
        violations = port_ret_pct < -var_curve
        n_viol = violations.sum()

        # Plotar
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(port_ret_pct.index, port_ret_pct, label="Retornos (%)", color="gray", alpha=0.6)
        ax.plot(var_curve.index, -var_curve, label=f"-VaR {int(confidence*100)}%", color="red", linewidth=2)
        ax.scatter(port_ret_pct.index[violations], port_ret_pct[violations], color="red", marker="x", s=40, label="Violações")
        ax.set_title(f"Curva Temporal do VaR ({metodo}) — {int(confidence*100)}%")
        ax.set_ylabel("Retornos (%)")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"🔴 **Número de violações:** {n_viol}  —  ({n_viol/len(var_curve)*100:.2f}% das observações)")

        # ========== Backtesting ==========
        st.subheader("🧪 Backtesting (Kupiec Test)")
        N, p_value = kupiec_test(port_ret_pct / 100, var_curve, confidence)
        st.write(f"Violações observadas: {N}")
        st.write(f"P-valor do teste: {p_value:.4f}")
        st.write("Adequação:", "✅ Sim" if p_value > 0.05 else "❌ Não")

        # ========== Download ==========
        buffer = io.BytesIO()
        df_curve = pd.DataFrame({"Retornos_%": port_ret_pct, f"VaR_{int(confidence*100)}_%": -var_curve})
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_curve.to_excel(writer, sheet_name="Curva_VaR", index=True)
        st.download_button(
            label="💾 Baixar Curva de VaR (Excel)",
            data=buffer.getvalue(),
            file_name="curva_var.xlsx",
            mime="application/vnd.ms-excel",
        )
