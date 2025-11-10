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
Ferramenta para calcular e comparar **Value at Risk (VaR)** de uma carteira de ações.
Inclui métodos:
- 📈 Histórico
- 📊 Paramétrico (Normal)
- 🎲 Monte Carlo

E também **teste de backtesting (Kupiec)** para validação estatística.
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

# ========= Interface principal =========
aba = st.tabs(["📈 Análise Individual", "⚖️ Comparativo de Métodos"])

# ========= Painel lateral comum =========
with st.sidebar:
    st.header("⚙️ Parâmetros Gerais")
    tickers = st.text_input("Tickers (separados por vírgula)", "VBBR3.SA, MCD, UBER, VALE3.SA, GS")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    start = st.date_input("Data inicial", pd.to_datetime("2022-01-01"))
    end = st.date_input("Data final", pd.Timestamp.today())
    confidence_levels = st.multiselect("Níveis de confiança", [0.95, 0.975, 0.99], default=[0.95, 0.99])
    horizon = st.number_input("Horizonte (dias)", min_value=1, value=1)
    usar_pesos = st.checkbox("Inserir pesos personalizados")

# ========= Função auxiliar para cálculo único =========
def calcular_var_por_metodo(rets, w, mu_vec, cov_mat, mu_p, sigma_p, metodo, alphas, n_sims):
    resultados = []
    for alpha in alphas:
        if metodo == "Histórico":
            var_val = var_historico(rets @ w, alpha)
        elif metodo == "Paramétrico (Normal)":
            var_val = var_parametrico(mu_p, sigma_p, alpha)
        else:
            var_val = var_mc(mu_vec, cov_mat, w, alpha, n_sims)
        resultados.append({"Confiança": alpha, "Método": metodo, "VaR_%": 100 * var_val})
    return pd.DataFrame(resultados)

# ========= Aba 1 — Análise Individual =========
with aba[0]:
    st.subheader("📈 Análise Individual de um Método")
    metodo = st.selectbox("Selecione o método de VaR", ["Histórico", "Paramétrico (Normal)", "Monte Carlo"])
    n_sims = st.number_input("Nº de Simulações (Monte Carlo)", 10_000, 200_000, 100_000, step=10_000)
    run = st.button("🚀 Calcular VaR (Individual)")

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

            df_var = calcular_var_por_metodo(rets, w, mu_vec, cov_mat, mu_p, sigma_p, metodo, confidence_levels, n_sims)
            st.dataframe(df_var.style.format(precision=3))

            # Gráfico
            fig, ax = plt.subplots()
            ax.bar(df_var["Confiança"].astype(str), df_var["VaR_%"], color="skyblue")
            ax.set_ylabel("VaR (% do patrimônio)")
            ax.set_title(f"VaR ({metodo})")
            st.pyplot(fig)

            # Backtesting
            st.markdown("### 🧪 Backtesting (Kupiec Test)")
            alpha = confidence_levels[0]
            var_hist_movel = -port_ret.rolling(252).quantile(1 - alpha).dropna() * 100
            rets_bt = port_ret.loc[var_hist_movel.index]
            N, p_value = kupiec_test(rets_bt, var_hist_movel, alpha)
            st.write(f"Confiança: {alpha:.1%} — Violações: {N}, P-valor: {p_value:.4f}")
            st.write("Adequação:", "✅ Sim" if p_value > 0.05 else "❌ Não")

# ========= Aba 2 — Comparativo =========
with aba[1]:
    st.subheader("⚖️ Comparativo entre Métodos")
    n_sims_cmp = st.number_input("Nº de Simulações (Monte Carlo)", 10_000, 200_000, 100_000, step=10_000, key="cmp")
    run_cmp = st.button("🚀 Calcular Comparativo")

    if run_cmp:
        result = get_data_and_returns(tickers, start, end)
        if result:
            rets, valid_tickers = result
            mu_vec, sigma_vec, cov_mat = rets.mean().values, rets.std().values, rets.cov().values
            w = np.array([1 / len(valid_tickers)] * len(valid_tickers))
            port_ret = rets @ w
            mu_p, sigma_p = port_ret.mean(), port_ret.std()

            df_hist = calcular_var_por_metodo(rets, w, mu_vec, cov_mat, mu_p, sigma_p, "Histórico", confidence_levels, n_sims_cmp)
            df_param = calcular_var_por_metodo(rets, w, mu_vec, cov_mat, mu_p, sigma_p, "Paramétrico (Normal)", confidence_levels, n_sims_cmp)
            df_mc = calcular_var_por_metodo(rets, w, mu_vec, cov_mat, mu_p, sigma_p, "Monte Carlo", confidence_levels, n_sims_cmp)
            df_all = pd.concat([df_hist, df_param, df_mc])

            st.markdown("### 📊 Tabela comparativa de VaR (%)")
            st.dataframe(df_all.pivot(index="Confiança", columns="Método", values="VaR_%").style.format(precision=3))

            # Gráfico comparativo
            fig, ax = plt.subplots()
            df_pivot = df_all.pivot(index="Confiança", columns="Método", values="VaR_%")
            df_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("VaR (% do patrimônio)")
            ax.set_title("Comparativo de Métodos de VaR")
            st.pyplot(fig)

            # Download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_hist.to_excel(writer, sheet_name="VaR_Histórico", index=False)
                df_param.to_excel(writer, sheet_name="VaR_Paramétrico", index=False)
                df_mc.to_excel(writer, sheet_name="VaR_MonteCarlo", index=False)
                df_all.to_excel(writer, sheet_name="Comparativo", index=False)
            st.download_button(
                label="💾 Baixar resultados (Excel)",
                data=buffer.getvalue(),
                file_name="comparativo_var.xlsx",
                mime="application/vnd.ms-excel",
            )

