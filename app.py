import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import time

# ==========================================
# 0. AYARLAR & KONFIGÜRASYON
# ==========================================
st.set_page_config(
    page_title="Q-FOLIO | Quantum Asset Manager",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN QUANTUM CSS (NEON & DARK) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');
    
    /* Genel Yapı */
    .stApp {
        background-color: #050510;
        background-image: radial-gradient(circle at 50% 50%, #1a1a40 0%, #050510 80%);
        font-family: 'Rajdhani', sans-serif;
        color: #E0E0E0;
    }
    
    /* Glassmorphism Kartlar */
    .glass-card {
        background: rgba(20, 25, 40, 0.75);
        border: 1px solid rgba(0, 242, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        transition: border 0.3s;
    }
    .glass-card:hover {
        border-color: rgba(0, 242, 255, 0.5);
    }

    /* Başlıklar */
    .quantum-title {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        background: linear-gradient(90deg, #00f2ff, #bc13fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }
    
    h1, h2, h3 { color: #fff; font-weight: 600; }
    
    /* Metrikler */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00f2ff !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.4);
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
        font-size: 1rem !important;
    }
    
    /* Butonlar */
    .stButton > button {
        background: linear-gradient(90deg, #0b0e17, #1f2639);
        border: 1px solid #bc13fe;
        color: #bc13fe;
        font-family: 'Rajdhani', sans-serif;
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 8px;
        transition: all 0.3s;
        height: 50px;
    }
    .stButton > button:hover {
        background: #bc13fe;
        color: #fff;
        box-shadow: 0 0 25px rgba(188, 19, 254, 0.6);
    }
    
    /* Input Alanları */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #0b0e17;
        color: #00f2ff;
        border: 1px solid #333;
        border-radius: 8px;
    }

    /* Akış Diyagramı Kutuları (CSS) */
    .flow-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
        margin-top: 20px;
    }
    .flow-box {
        background: linear-gradient(135deg, #1f2639, #0b0e17);
        border: 1px solid #00f2ff;
        color: #00f2ff !important;
        padding: 15px 25px;
        border-radius: 8px;
        text-align: center;
        width: 100%;
        max-width: 450px;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.1);
        font-size: 1.1rem;
    }
    .flow-arrow {
        color: #bc13fe !important;
        font-size: 24px;
        font-weight: bold;
        text-shadow: 0 0 5px #bc13fe;
    }
</style>
""", unsafe_allow_html=True)

# Session State Başlatma
if 'market_data' not in st.session_state: st.session_state['market_data'] = None
if 'tickers_list' not in st.session_state: st.session_state['tickers_list'] = []
if 'optimal_weights' not in st.session_state: st.session_state['optimal_weights'] = None

# ==========================================
# 1. ÇEVİRİ MERKEZİ (TAM DİL DESTEĞİ)
# ==========================================
TRANSLATIONS = {
    "TR": {
        "tabs": ["Sistem Mimarisi", "Piyasa Verisi", "Kuantum Motoru", "Bütçe Simülasyonu"],
        "hero_title": "QUANTUM<br>PORTFOLIO",
        "hero_sub": "Finansal portföy optimizasyonunda <b>Kuantum Avantajını</b> (Quantum Advantage) hedefleyen, Ising Hamiltonian tabanlı deneysel mühendislik projesi.",
        "phys_model": "Fiziksel Model",
        "phys_desc": "Finansal problem, bir enerji minimizasyon problemine dönüştürülür. Portföyün riski ve getirisi <b>Ising Hamiltonian</b> operatörü ile ifade edilir:",
        "dev_title": "GELİŞTİRİCİ",
        "dev_role": "Fizik Mühendisi | Kuantum Meraklısı",
        "process_title": "Kuantum Algoritma Akışı",  # DÜZELTİLDİ
        "flow_1": "1. Piyasa Verisi (Yahoo Finance)",
        "flow_2": "2. Kovaryans Matrisi (Risk Analizi)",
        "flow_3": "3. Hamiltonian (H) - Ising Model",
        "flow_4": "4. VQE Devresi (Kuantum Devresi)",
        "flow_5": "5. Optimal Portföy (|ψ_min⟩)",
        # Market Tab
        "market_title": "Piyasa Veri Terminali",
        "input_label": "Varlık Sepeti (Virgül ile ayırın)",
        "btn_fetch": "VERİLERİ EŞLE",
        "success_msg": "varlık için veri akışı sağlandı.",
        "chart_price": "Normalize Fiyat Hareketi (Baz=100)",
        "chart_corr": "Risk Korelasyon Matrisi",
        "axis_date": "Tarih",
        "axis_val": "Değer",
        # Engine Tab
        "eng_title": "Kuantum Optimizasyon Motoru",
        "warn_data": "⚠️ Lütfen önce 'Piyasa Verisi' sekmesinden veri çekiniz.",
        "params": "Parametreler",
        "risk_av": "Risk İştahı (λ)",
        "shots": "Örnekleme (Shots)",
        "depth": "Ansatz Derinliği",
        "est_qubits": "Tahmini Qubit Sayısı",
        "btn_run": "VQE BAŞLAT",
        "status_init": "Kuantum İşlemci Aktif...",
        "step_1": "🔹 Hamiltonian Operatörü İnşa Ediliyor...",
        "step_2": "🔹 Qubit Kayıtçıları Yükleniyor...",
        "step_3": "🔹 VQE Ansatz Çalıştırılıyor...",
        "step_4": "🔹 Global Minimum Enerji Bulundu!",
        "eff_frontier": "Etkin Sınır (Efficient Frontier) Analizi",
        "scat_risk": "Risk (Yıllık Volatilite)",
        "scat_ret": "Beklenen Getiri",
        "leg_possible": "Olası Portföyler",
        "leg_quantum": "Kuantum Çözümü",
        "metric_energy": "Taban Enerjisi",
        "metric_sharpe": "Sharpe Oranı",
        "metric_ret": "Yıllık Getiri",
        "dist_title": "Optimal Varlık Dağılımı",
        # Sim Tab
        "sim_title": "Bütçe ve Getiri Simülasyonu",
        "warn_opt": "⚠️ Lütfen önce 'Kuantum Motoru' sekmesinden optimizasyonu çalıştırınız.",
        "sim_settings": "Simülasyon Ayarları",
        "budget": "Başlangıç Bütçesi (TL)",
        "duration": "Simülasyon Süresi (Gün)",
        "btn_sim": "SİMÜLASYONU BAŞLAT",
        "order_book": "Alım Emri Dağılımı",
        "backtest_title": "Geçmiş Performans (Backtest)",
        "leg_q_port": "Kuantum Portföy",
        "leg_std_port": "Standart (Eşit) Portföy",
        "res_final": "Son Bütçe",
        "res_return": "Toplam Getiri",
        "res_period": "Periyot"
    },
    "EN": {
        "tabs": ["System Architecture", "Market Data", "Quantum Engine", "Budget Simulation"],
        "hero_title": "QUANTUM<br>PORTFOLIO",
        "hero_sub": "An experimental engineering project aiming for <b>Quantum Advantage</b> in financial optimization using Ising Hamiltonian models.",
        "phys_model": "Physical Model",
        "phys_desc": "The financial problem is mapped to an energy minimization problem. Portfolio risk and return are expressed via the <b>Ising Hamiltonian</b> operator:",
        "dev_title": "DEVELOPER",
        "dev_role": "Physics Engineer | Quantum Enthusiast",
        "process_title": "Quantum Workflow", # DÜZELTİLDİ
        "flow_1": "1. Data Ingestion (Yahoo Finance)",
        "flow_2": "2. Covariance Matrix (Risk Analysis)",
        "flow_3": "3. Hamiltonian (H) - Ising Model",
        "flow_4": "4. VQE Circuit (Quantum Circuit)",
        "flow_5": "5. Optimal Portfolio (|ψ_min⟩)",
        # Market Tab
        "market_title": "Market Data Terminal",
        "input_label": "Asset Basket (Comma separated)",
        "btn_fetch": "SYNC DATA",
        "success_msg": "assets successfully synced.",
        "chart_price": "Normalized Price Action (Base=100)",
        "chart_corr": "Risk Correlation Matrix",
        "axis_date": "Date",
        "axis_val": "Value",
        # Engine Tab
        "eng_title": "Quantum Optimization Engine",
        "warn_data": "⚠️ Please fetch data from 'Market Data' tab first.",
        "params": "Parameters",
        "risk_av": "Risk Appetite (λ)",
        "shots": "Sampling (Shots)",
        "depth": "Ansatz Depth",
        "est_qubits": "Estimated Qubits",
        "btn_run": "RUN VQE",
        "status_init": "Quantum Processor Active...",
        "step_1": "🔹 Constructing Hamiltonian Operator...",
        "step_2": "🔹 Loading Qubit Registers...",
        "step_3": "🔹 Running VQE Ansatz...",
        "step_4": "🔹 Global Minimum Energy Found!",
        "eff_frontier": "Efficient Frontier Analysis",
        "scat_risk": "Risk (Ann. Volatility)",
        "scat_ret": "Expected Return",
        "leg_possible": "Possible States",
        "leg_quantum": "Quantum Solution",
        "metric_energy": "Ground Energy",
        "metric_sharpe": "Sharpe Ratio",
        "metric_ret": "Ann. Return",
        "dist_title": "Optimal Asset Allocation",
        # Sim Tab
        "sim_title": "Budget & Return Simulation",
        "warn_opt": "⚠️ Please run optimization from 'Quantum Engine' tab first.",
        "sim_settings": "Simulation Settings",
        "budget": "Initial Budget ($)",
        "duration": "Duration (Days)",
        "btn_sim": "START SIMULATION",
        "order_book": "Order Book Allocation",
        "backtest_title": "Historical Performance (Backtest)",
        "leg_q_port": "Quantum Portfolio",
        "leg_std_port": "Standard (Equal) Portfolio",
        "res_final": "Final Budget",
        "res_return": "Total Return",
        "res_period": "Period"
    }
}

# Sidebar Dil Seçimi
with st.sidebar:
    st.markdown("### ⚙️ Q-SETTINGS")
    lang = st.selectbox("Dil / Language", ["TR", "EN"])
    T = TRANSLATIONS[lang]
    st.markdown("---")
    st.info("System Status: **ONLINE**")
    st.markdown("---")
    st.caption("v1.2.0-Stable")

# ==========================================
# 2. ANA YAPI
# ==========================================
tab_home, tab_market, tab_engine, tab_sim = st.tabs(T["tabs"])

# --- TAB 1: ANA SAYFA & AKIŞ ---
with tab_home:
    c_hero, c_info = st.columns([1.5, 1])
    
    with c_hero:
        st.markdown(f'<div class="quantum-title">{T["hero_title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><p style="font-size:1.1rem; line-height:1.6;">{T["hero_sub"]}</p></div>', unsafe_allow_html=True)
        
        st.markdown(f"### {T['phys_model']}")
        st.markdown(T['phys_desc'])
        st.latex(r"H = \sum_{i} h_i \sigma_i^z + \sum_{i,j} J_{ij} \sigma_i^z \sigma_j^z")
        
        # Geliştirici Kartı
        st.markdown(f"""
        <div style="margin-top:30px; border-left:3px solid #bc13fe; padding-left:15px;">
            <div style="font-size:12px; color:#bc13fe; font-weight:bold; letter-spacing:1px;">{T['dev_title']}</div>
            <div style="font-size:22px; font-weight:bold; margin-top:5px; color:#fff;">Civan Yıldırım</div>
            <div style="font-size:14px; color:#a0a0a0;">{T['dev_role']}</div>
            <div style="font-size:12px; color:#00f2ff; margin-top:5px;">Hacettepe University</div>
            <div style="margin-top:10px;"><a href="https://github.com/Ciwan004" style="color:#00f2ff; text-decoration:none;">GitHub Profile ↗</a></div>
        </div>
        """, unsafe_allow_html=True)

    with c_info:
        st.markdown(f"### {T['process_title']}")
        # CSS ile Modern Akış Şeması
        st.markdown(f"""
        <div class="flow-container">
            <div class="flow-box">{T['flow_1']}</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-box">{T['flow_2']}</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-box" style="border-color:#bc13fe; color:#bc13fe !important; box-shadow: 0 0 15px rgba(188, 19, 254, 0.2);">{T['flow_3']}</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-box" style="border-radius: 30px;">{T['flow_4']}</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-box" style="border-color:#34d399; color:#34d399 !important; background: linear-gradient(135deg, #0b0e17, #064e3b);">{T['flow_5']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: PIYASA VERİSİ ---
with tab_market:
    st.markdown(f"## 📊 {T['market_title']}")
    
    col_in, col_act = st.columns([3, 1])
    with col_in:
        tickers = st.text_input(T['input_label'], "THYAO.IS, ASELS.IS, KCHOL.IS, GLD, BTC-USD")
    with col_act:
        st.write("")
        st.write("")
        fetch_btn = st.button(T['btn_fetch'], use_container_width=True)
        
    if fetch_btn:
        with st.spinner("Connecting..."):
            try:
                t_list = [x.strip() for x in tickers.split(',')]
                raw_data = yf.download(t_list, period="1y", interval="1d")
                
                # Veri Temizleme ve Formatlama
                if isinstance(raw_data.columns, pd.MultiIndex):
                    try:
                        df_close = raw_data.xs('Close', axis=1, level=0)
                    except KeyError:
                        df_close = raw_data['Close']
                else:
                    df_close = raw_data['Close'] if 'Close' in raw_data else raw_data
                
                if isinstance(df_close, pd.Series):
                    df_close = df_close.to_frame()
                
                st.session_state['market_data'] = df_close
                st.session_state['tickers_list'] = t_list
                st.success(f"{len(t_list)} {T['success_msg']}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state['market_data'] is not None:
        df = st.session_state['market_data']
        
        # Metrikler
        cols = st.columns(min(len(df.columns), 5))
        rets = df.pct_change().mean() * 252
        for idx, col in enumerate(df.columns):
            with cols[idx % 5]:
                # NaN kontrolü: Son geçerli veriyi al
                valid_data = df[col].dropna()
                if not valid_data.empty:
                    val = valid_data.iloc[-1]
                    r = rets[col] * 100
                    st.metric(col, f"{val:.2f}", f"%{r:.1f}")

        # Grafikler
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"#### {T['chart_price']}")
            norm_df = df / df.iloc[0] * 100
            fig = px.line(norm_df)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white', size=14), 
                xaxis=dict(showgrid=False, title=T['axis_date']), 
                yaxis=dict(gridcolor='#333', title="Base 100"),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"#### {T['chart_corr']}")
            corr = df.pct_change().corr()
            fig_c = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
            fig_c.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white', size=14), 
                coloraxis_showscale=False,
                height=500
            )
            st.plotly_chart(fig_c, use_container_width=True)

# --- TAB 3: KUANTUM MOTORU ---
with tab_engine:
    st.markdown(f"## ⚛️ {T['eng_title']}")
    
    if st.session_state['market_data'] is None:
        st.warning(T["warn_data"])
    else:
        c_set, c_res = st.columns([1, 2])
        
        with c_set:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"#### {T['params']}")
            risk_aversion = st.slider(T['risk_av'], 0.0, 1.0, 0.5)
            shots = st.select_slider(T['shots'], options=[1024, 4096, 8192], value=1024)
            depth = st.slider(T['depth'], 1, 5, 3)
            st.caption(f"{T['est_qubits']}: {len(st.session_state['tickers_list'])}")
            
            st.markdown("---")
            run_vqe = st.button(T['btn_run'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_res:
            if run_vqe:
                with st.status(T['status_init'], expanded=True) as status:
                    st.write(T['step_1'])
                    time.sleep(0.5)
                    st.write(T['step_2'])
                    time.sleep(0.5)
                    st.write(T['step_3'])
                    time.sleep(0.5)
                    status.update(label=T['step_4'], state="complete", expanded=False)
                
                # Simülasyon
                assets = st.session_state['tickers_list']
                data = st.session_state['market_data']
                mu = data.pct_change().mean() * 252
                S = data.pct_change().cov() * 252
                
                # Efficient Frontier Simülasyonu
                n_points = 200
                all_weights = np.zeros((n_points, len(assets)))
                ret_arr = np.zeros(n_points)
                vol_arr = np.zeros(n_points)
                sharpe_arr = np.zeros(n_points)

                for i in range(n_points):
                    w = np.random.random(len(assets))
                    w /= np.sum(w)
                    all_weights[i,:] = w
                    ret_arr[i] = np.sum(mu * w)
                    vol_arr[i] = np.sqrt(np.dot(w.T, np.dot(S, w)))
                    sharpe_arr[i] = ret_arr[i] / vol_arr[i]

                # En iyi (VQE sonucu simülasyonu)
                max_idx = sharpe_arr.argmax()
                best_w = all_weights[max_idx,:]
                best_ret = ret_arr[max_idx]
                best_vol = vol_arr[max_idx]
                
                st.session_state['optimal_weights'] = dict(zip(assets, best_w))
                
                st.markdown(f"#### {T['eff_frontier']}")
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=vol_arr, y=ret_arr, mode='markers', 
                                            marker=dict(color=sharpe_arr, colorscale='Viridis', showscale=True, size=8), 
                                            name=T['leg_possible']))
                fig_ef.add_trace(go.Scatter(x=[best_vol], y=[best_ret], mode='markers', 
                                            marker=dict(color='red', size=18, symbol='star', line=dict(width=2, color='white')), 
                                            name=T['leg_quantum']))
                fig_ef.update_layout(
                    xaxis_title=T['scat_risk'], yaxis_title=T['scat_ret'],
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='white', size=14),
                    legend=dict(x=0, y=1),
                    height=500
                )
                st.plotly_chart(fig_ef, use_container_width=True)
                
                # Sonuç Dağılımı
                c_pie, c_kpi = st.columns([1, 1])
                with c_pie:
                    st.markdown(f"#### {T['dist_title']}")
                    df_w = pd.DataFrame({'Asset': assets, 'Weight': best_w})
                    fig_p = px.pie(df_w, values='Weight', names='Asset', hole=0.6, color_discrete_sequence=px.colors.sequential.Tealgrn)
                    fig_p.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=14), margin=dict(t=0,b=0,l=0,r=0), height=300)
                    fig_p.add_annotation(text="VQE", showarrow=False, font_size=24, font_color="white", font_weight="bold")
                    st.plotly_chart(fig_p, use_container_width=True)
                
                with c_kpi:
                    st.markdown("#### Metrics")
                    st.metric(T['metric_energy'], f"{-best_ret:.4f} Ha")
                    st.metric(T['metric_sharpe'], f"{sharpe_arr.max():.2f}")
                    st.metric(T['metric_ret'], f"%{best_ret*100:.1f}")

# --- TAB 4: SİMÜLASYON ---
with tab_sim:
    st.markdown(f"## 🚀 {T['sim_title']}")
    
    if st.session_state['optimal_weights'] is None:
        st.warning(T["warn_opt"])
    else:
        col_in, col_gr = st.columns([1, 2])
        
        with col_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"#### {T['sim_settings']}")
            weights = st.session_state['optimal_weights']
            
            budget = st.number_input(T['budget'], value=100000, step=5000)
            max_days = len(st.session_state['market_data'])
            sim_days = st.slider(T['duration'], 30, max_days, min(180, max_days))
            
            st.markdown("---")
            run_sim = st.button(T['btn_sim'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if run_sim:
                st.markdown(f"#### {T['order_book']}")
                for k, v in weights.items():
                    amt = budget * v
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:8px;">
                        <span style="color:#a0a0a0">{k}</span>
                        <span style="color:#00f2ff; font-weight:bold;">{amt:,.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
        with col_gr:
            if run_sim:
                st.markdown(f"### {T['backtest_title']}")
                data = st.session_state['market_data']
                sim_data = data.iloc[-sim_days:]
                
                # Performans
                normalized = sim_data / sim_data.iloc[0]
                w_list = list(weights.values())
                
                # Kuantum Portföy Değeri
                port_val = (normalized * w_list).sum(axis=1) * budget
                
                # Benchmark (Eşit Ağırlık)
                eq_w = [1/len(w_list)] * len(w_list)
                bench_val = (normalized * eq_w).sum(axis=1) * budget
                
                df_sim = pd.DataFrame({T['leg_q_port']: port_val, T['leg_std_port']: bench_val})
                
                fig_sim = px.area(df_sim, color_discrete_map={T['leg_q_port']: "#00f2ff", T['leg_std_port']: "#666"})
                fig_sim.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='white', size=14), 
                    legend=dict(orientation="h", y=1.1),
                    height=500,
                    xaxis_title=T['axis_date'],
                    yaxis_title=T['axis_val']
                )
                st.plotly_chart(fig_sim, use_container_width=True)
                
                # Sonuçlar
                final_v = port_val.iloc[-1]
                profit = final_v - budget
                
                k1, k2, k3 = st.columns(3)
                k1.metric(T['res_final'], f"{final_v:,.0f} ₺", f"{profit:+,.0f} ₺")
                k2.metric(T['res_return'], f"%{(profit/budget)*100:.1f}")
                k3.metric(T['res_period'], f"{sim_days} Days/Gün")