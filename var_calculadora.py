# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import io

# ========= Configuração =========
st.set_page_config(page_title="Painel VaR - Três Métodos", layout="wide")
st.title("📊 Painel Interativo de Value at Risk (VaR) — Comparação Completa")

st.markdown("""
Este painel calcula, compara e mostra a **curva temporal do VaR** de uma carteira com três métodos:
- 📈 **Histórico**
- 📊 **Paramétrico (Normal)**
- 🎲 **Monte Carlo**

Inclui **backtesting de Kupiec**, **gráficos dinâmicos** e **download dos resultados**.
""")

# ========= Funções base =========
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

def var_mc(mu_vec, cov, w, alpha, n_sims=100_000):
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

# ========= Parâmetros =========
with st.sidebar:
    st.header("⚙️ Parâmetros da Análise")
    tickers = st.text_input("Tickers (separados por vírgula)", "VBBR3.SA, MCD, UBER, VALE3.SA, GS")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    start = st.date_input("Data inicial", pd.to_datetime("2020-01-01"))
    end = st.date_input("Data final", pd.Timestamp.today())
    confidence = st.selectbox("Nível de confiança", [0.95, 0.975, 0.99], index=0)
    n_sims = st.number_input("Simulações (Monte Carlo)", 10_000, 200_000, 100_000, step=10_000)
    usar_pesos = st.checkbox("Inserir pesos personalizados")
    run = st.button("🚀 Rodar Análise Completa")

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

        # ========= VaR Estático =========
        vh = var_historico(port_ret, confidence)
        vp = var_parametrico(mu_p, sigma_p, confidence)
        vm = var_mc(mu_vec, cov_mat, w, confidence, n_sims)

        st.subheader("📉 VaR Estático da Carteira")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Histórico ({int(confidence*100)}%)", f"{100*vh:.3f}%")
        col2.metric(f"Paramétrico ({int(confidence*100)}%)", f"{100*vp:.3f}%")
        col3.metric(f"Monte Carlo ({int(confidence*100)}%)", f"{100*vm:.3f}%")

        # ========= Curvas Temporais =========
        st.subheader("📊 Curvas Temporais do VaR e Retornos")
        window = 252

        # Retornos
        port_ret_pct = port_ret * 100

        # Histórico
        var_hist = -port_ret.rolling(window).quantile(1 - confidence).dropna() * 100
        # Paramétrico
        mu_roll = port_ret.rolling(window).mean()
        sigma_roll = port_ret.rolling(window).std()
        z = norm.ppf(1 - confidence)
        var_param = -(mu_roll + z * sigma_roll).dropna() * 100
        # Monte Carlo aproximado (via covariância)
        mc_curve = []
        for i in range(window, len(port_ret)):
            mu_local = rets.iloc[i-window:i].mean().values
            cov_local = rets.iloc[i-window:i].cov().values
            var_val = var_mc(mu_local, cov_local, w, confidence, int(n_sims/50))  # reduzir para velocidade
            mc_curve.append(var_val*100)
        var_mc_curve = pd.Series(mc_curve, index=port_ret.index[window:])

        # ========== Gráfico ==========
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(port_ret_pct.index, port_ret_pct, color="gray", alpha=0.5, label="Retornos (%)")
        ax.plot(var_hist.index, -var_hist, label="VaR Histórico", color="blue", linewidth=1.8)
        ax.plot(var_param.index, -var_param, label="VaR Paramétrico", color="orange", linewidth=1.8)
        ax.plot(var_mc_curve.index, -var_mc_curve, label="VaR Monte Carlo", color="green", linewidth=1.8)
        ax.set_title(f"Curvas de VaR ({int(confidence*100)}%) — Janela {window} dias")
        ax.set_ylabel("Retorno / VaR (%)")
        ax.legend()
        st.pyplot(fig)

        # ========= Backtesting =========
        st.subheader("🧪 Backtesting (Kupiec Test)")
        results_bt = []
        for metodo, var_series in [("Histórico", var_hist), ("Paramétrico", var_param), ("Monte Carlo", var_mc_curve)]:
            aligned = port_ret.loc[var_series.index]
            N, p_value = kupiec_test(aligned, var_series, confidence)
            results_bt.append({
                "Método": metodo,
                "Violações": N,
                "P-valor": p_value,
                "Adequado": "Sim" if p_value > 0.05 else "Não"
            })
        df_bt = pd.DataFrame(results_bt)
        st.dataframe(df_bt.style.format(precision=4))

        # ========= Download =========
        st.subheader("💾 Download dos Resultados")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"Retornos_%": port_ret_pct}).to_excel(writer, sheet_name="Retornos", index=True)
            var_hist.to_excel(writer, sheet_name="VaR_Histórico")
            var_param.to_excel(writer, sheet_name="VaR_Paramétrico")
            var_mc_curve.to_excel(writer, sheet_name="VaR_MonteCarlo")
            df_bt.to_excel(writer, sheet_name="Backtesting", index=False)
        st.download_button(
            label="⬇️ Baixar Resultados (Excel)",
            data=buffer.getvalue(),
            file_name="analise_var_completa.xlsx",
            mime="application/vnd.ms-excel",
        )
