#import streamlit as pd
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import config

st.set_page_config(page_title="IOVIQUANT Pro - Advanced", layout="wide")
st.title("⚡ IOVIQUANT Pro: Advanced Testing & Deep Dive")

# --- CALCOLO DATE DI DEFAULT (1 Gennaio Anno In Corso -> Oggi) ---
current_year = datetime.date.today().year
default_start = pd.to_datetime(f"{current_year}-01-01")
default_end = pd.to_datetime("today") #- pd.Timedelta(days=1)

# --- UI PARAMETRI (Sidebar) ---
with st.sidebar.form("backtest_params"):
    st.header("Parametri Core")
    start_date = st.date_input("Data Inizio", default_start)
    end_date = st.date_input("Data Fine", default_end)
    
    st.header("Money Management")
    initial_capital = st.number_input("Capitale Iniziale (€)", value=10000.0, step=1000.0)
    base_pos = st.number_input("Posizione Base (€)", value=200.0, step=50.0)
    hmm_w = st.slider("Peso Amplificatore HMM", 0.0, 3.0, 1.0, 0.1)
    
    st.header("Risk Management")
    sl_pct = st.slider("Stop Loss (% sotto EMA63)", 1.0, 10.0, 2.5, 0.1) / 100.0
    
    run_btn = st.form_submit_button("Elabora Dati e Backtest 🚀")

# --- MOTORE DATI ---
@st.cache_data(show_spinner=False)
def fetch_and_compute(tickers, start, end, hmm_weight):
    start_offset = pd.to_datetime(start) - pd.Timedelta(days=365)
    
    raw_tickers = yf.download(tickers, start=start_offset, end=end, progress=False)
    raw_macro = yf.download(['^VIX', config.BENCHMARK], start=start_offset, end=end, progress=False)
    
    vix = raw_macro['Close']['^VIX'].ffill()
    bench = raw_macro['Close'][config.BENCHMARK].ffill()
    close_prices = raw_tickers['Close'].ffill()
    
    sma200 = close_prices.rolling(200).mean()
    breadth = (close_prices > sma200).mean(axis=1) * 100 
    
    data_dict = {}
    
    for ticker in tickers:
        try:
            df = raw_tickers.xs(ticker, level=1, axis=1).dropna() if len(tickers) > 1 else raw_tickers.dropna()
            if len(df) < 100: continue
            
            df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['EMA63'] = df['Close'].ewm(span=63, adjust=False).mean()
            
            df['Buy_Tech'] = (df['EMA5'] > df['EMA21']) & (df['EMA21'] > df['EMA63'])
            df['Sell_Tech'] = (df['EMA5'] < df['EMA21']) & (df['EMA21'] < df['EMA63'])
            
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
            features = df[['Log_Ret']]
            
            hmm = GaussianHMM(n_components=2, covariance_type="full", random_state=42).fit(features)
            probs = hmm.predict_proba(features)
            bull_state = np.argmax(hmm.means_)
            df['P_Bull'] = probs[:, bull_state]
            
            df['Signal_Mult'] = df['Buy_Tech'].astype(int) * (1 + hmm_weight * (df['P_Bull'] - 0.5))
            
            data_dict[ticker] = df[['Close', 'EMA5', 'EMA21', 'EMA63', 'Buy_Tech', 'Sell_Tech', 'Signal_Mult', 'P_Bull']]
        except:
            pass
            
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    def filter_by_date(series):
        idx = series.index.tz_localize(None) if series.index.tz is not None else series.index
        mask = (idx >= start_dt) & (idx <= end_dt)
        return series.loc[mask]

    filtered_data_dict = {k: filter_by_date(v) for k, v in data_dict.items()}

    return filtered_data_dict, filter_by_date(vix), filter_by_date(bench), filter_by_date(breadth)

# --- MOTORE BACKTEST EVENT-DRIVEN ---
def run_simulation(data_dict, target_tickers, capital, base_size, sl_factor, vix_series, breadth_series):
    cash = capital
    portfolio = {}
    history = []
    
    events = [] 
    closed_trades = [] 
    FEE = 1.0 
    
    all_dates = pd.concat([df.index.to_series() for df in data_dict.values()]).unique()
    dates = sorted(all_dates)
    
    for date in dates:
        current_vix = vix_series.get(date, 20)
        current_breadth = breadth_series.get(date, 50)
        
        halt_trading = current_breadth < 40 and len(target_tickers) > 1
        size_mult_vix = 0.5 if current_vix > 25 else 1.0
        
        # 1. GESTIONE VENDITE (Uscite)
        tickers_to_remove = []
        for t, pos in portfolio.items():
            if date in data_dict[t].index:
                row = data_dict[t].loc[date]
                price = float(row['Close'])
                
                is_sl = price < (pos['ema63'] * sl_factor)
                is_signal_sell = bool(row['Sell_Tech'])
                
                if is_sl or is_signal_sell:
                    revenue = (pos['shares'] * price) - FEE
                    cash += revenue
                    profit = revenue - pos['cost_basis']
                    
                    motivo = "SL" if is_sl else "Signal"
                    
                    closed_trades.append({
                        'Entry_Date': pos['entry_date'],
                        'Exit_Date': date,
                        'Ticker': t,
                        'Shares': pos['shares'],
                        'Buy_Price': pos['buy_price'],
                        'Sell_Price': price,
                        'Signal_Strength': pos['signal_mult'],
                        'Entry_VIX': pos['entry_vix'],
                        'Entry_Breadth': pos['entry_breadth'],
                        'Profit': profit,
                        'Motivo_Chiusura': motivo,
                        'Stato': 'Chiusa'
                    })
                    events.append({'Date': date, 'Ticker': t, 'Action': 'SELL', 'Price': price})
                    tickers_to_remove.append(t)
                else:
                    portfolio[t]['ema63'] = float(row['EMA63'])
                    
        for t in tickers_to_remove:
            del portfolio[t]

        # 2. GESTIONE ACQUISTI
        if not halt_trading:
            daily_signals = []
            for t in target_tickers:
                if t in data_dict and date in data_dict[t].index and t not in portfolio:
                    row = data_dict[t].loc[date]
                    if bool(row['Buy_Tech']) and float(row['Signal_Mult']) > 0:
                        daily_signals.append({'ticker': t, 'price': float(row['Close']), 'mult': float(row['Signal_Mult'])})
            
            daily_signals = sorted(daily_signals, key=lambda x: x['mult'], reverse=True)
            
            for sig in daily_signals:
                target_size = base_size * sig['mult'] * size_mult_vix
                target_size = min(target_size, capital * 0.15) 
                
                cost_to_buy = target_size + FEE
                if cash >= cost_to_buy:
                    shares = int(target_size / sig['price'])
                    if shares > 0:
                        actual_cost = (shares * sig['price']) + FEE
                        cash -= actual_cost
                        
                        portfolio[sig['ticker']] = {
                            'shares': shares, 
                            'ema63': float(data_dict[sig['ticker']].loc[date, 'EMA63']),
                            'entry_date': date,
                            'buy_price': sig['price'],
                            'cost_basis': actual_cost,
                            'signal_mult': sig['mult'],
                            'entry_vix': current_vix,
                            'entry_breadth': current_breadth
                        }
                        events.append({'Date': date, 'Ticker': sig['ticker'], 'Action': 'BUY', 'Price': sig['price']})

        # 3. VALUTAZIONE MARK-TO-MARKET
        daily_equity = cash
        for t, pos in portfolio.items():
            if date in data_dict[t].index:
                daily_equity += pos['shares'] * float(data_dict[t].loc[date, 'Close'])
        
        history.append({'Date': date, 'Equity': daily_equity})
        
    # Posizioni Aperte a Fine Periodo
    open_trades = []
    if len(dates) > 0:
        last_date = dates[-1]
        for t, pos in portfolio.items():
            if last_date in data_dict[t].index:
                curr_price = float(data_dict[t].loc[last_date, 'Close'])
                unrealized_profit = (pos['shares'] * curr_price) - FEE - pos['cost_basis']
                open_trades.append({
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': pd.NaT,
                    'Ticker': t,
                    'Shares': pos['shares'],
                    'Buy_Price': pos['buy_price'],
                    'Sell_Price': curr_price,
                    'Signal_Strength': pos['signal_mult'],
                    'Entry_VIX': pos['entry_vix'],
                    'Entry_Breadth': pos['entry_breadth'],
                    'Profit': unrealized_profit,
                    'Motivo_Chiusura': '-',
                    'Stato': 'Aperta'
                })

    df_closed = pd.DataFrame(closed_trades)
    df_open = pd.DataFrame(open_trades)
    df_all_trades = pd.concat([df_closed, df_open], ignore_index=True) if not df_closed.empty or not df_open.empty else pd.DataFrame()
        
    return pd.DataFrame(history).set_index('Date'), pd.DataFrame(events), df_all_trades, df_closed

def calc_metrics(series):
    ret = series.pct_change().fillna(0)
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1] / series.iloc[0]) ** (365.25 / days) - 1 if days > 0 else 0
    vol = ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    mdd = ((series - series.cummax()) / series.cummax()).min()
    return cagr, sharpe, mdd, series.iloc[-1] - series.iloc[0]

def render_audit_log(df_trades):
    if df_trades.empty:
        st.warning("Nessuna operazione registrata in questo intervallo di tempo.")
        return
        
    display_df = df_trades.copy()
    display_df['Entry_Date'] = display_df['Entry_Date'].dt.strftime('%Y-%m-%d')
    display_df['Exit_Date'] = display_df['Exit_Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else "-")
    display_df = display_df.sort_values(by='Entry_Date', ascending=False)
    
    # Riordino colonne con colonna 'Shares' inserita tra Ticker e Buy_Price
    column_order = ['Entry_Date', 'Exit_Date', 'Ticker', 'Shares', 'Buy_Price', 'Sell_Price', 'Signal_Strength', 'Entry_VIX', 'Entry_Breadth', 'Profit', 'Motivo_Chiusura', 'Stato']
    display_df = display_df[column_order]
    
    st.dataframe(
        display_df.style.format({
            'Shares': '{:,.0f}',
            'Buy_Price': '€ {:.2f}',
            'Sell_Price': '€ {:.2f}',
            'Profit': '€ {:.2f}',
            'Signal_Strength': '{:.2f}',
            'Entry_VIX': '{:.1f}',
            'Entry_Breadth': '{:.1f}%'
        }).map(lambda x: 'color: #00CC96' if x == 'Aperta' else ('color: #EF553B' if x == 'SL' else ('color: #636EFA' if x == 'Signal' else 'color: gray')), subset=['Stato', 'Motivo_Chiusura']),
        use_container_width=True
    )

# --- GESTIONE STATO E FLUSSO ---
if run_btn:
    with st.spinner("Elaborazione Universo Storico..."):
        st.session_state.data_dict, st.session_state.vix, st.session_state.bench, st.session_state.breadth = fetch_and_compute(
            config.UNIVERSE, start_date, end_date, hmm_w
        )
        st.session_state.run_done = True

# --- RENDERING DASHBOARD ---
if st.session_state.get('run_done', False):
    data_dict = st.session_state.data_dict
    vix = st.session_state.vix
    bench = st.session_state.bench
    breadth = st.session_state.breadth
    sl_multiplier = 1.0 - sl_pct

    tab1, tab2 = st.tabs(["📊 Portafoglio Globale", "🔍 Deep Dive (Singolo Titolo)"])
    
    # ==========================================
    # TAB 1: PORTAFOGLIO GLOBALE
    # ==========================================
    with tab1:
        hist_global, _, all_trades_global, closed_global = run_simulation(data_dict, config.UNIVERSE, initial_capital, base_pos, sl_multiplier, vix, breadth)
        
        bench_eq = (bench / bench.iloc[0]) * initial_capital
        res_global = hist_global.join(bench_eq.rename('VWCE_Eq')).ffill()
        
        cg_s, sh_s, md_s, nt_s = calc_metrics(res_global['Equity'])
        cg_b, sh_b, md_b, nt_b = calc_metrics(res_global['VWCE_Eq'])
        
        cols = st.columns(4)
        cols[0].metric("Net Profit Strat", f"€ {nt_s:,.2f}", f"vs € {nt_b:,.2f} (Bench)")
        cols[1].metric("CAGR", f"{cg_s:.2%}", f"{(cg_s - cg_b):+.2%} Alpha")
        cols[2].metric("Sharpe", f"{sh_s:.2f}", f"Bench: {sh_b:.2f}")
        cols[3].metric("Max Drawdown", f"{md_s:.2%}", f"Bench: {md_b:.2%}")
        
        fig_g1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_g1.add_trace(go.Scatter(x=res_global.index, y=res_global['Equity'], name='IOVIQUANT', line=dict(color='#00CC96')), secondary_y=False)
        fig_g1.add_trace(go.Scatter(x=res_global.index, y=res_global['VWCE_Eq'], name='VWCE', line=dict(color='#636EFA', dash='dot')), secondary_y=False)
        fig_g1.add_trace(go.Scatter(x=vix.index, y=vix, name='VIX', line=dict(color='rgba(255, 0, 0, 0.4)')), secondary_y=True)
        fig_g1.add_trace(go.Scatter(x=breadth.index, y=breadth, name='Breadth %', line=dict(color='#FFA500', width=1.5)), secondary_y=True)
        fig_g1.update_layout(title="Equity Line Globale & Macro", template="plotly_dark", height=400)
        st.plotly_chart(fig_g1, use_container_width=True)
        
        st.markdown("---")
        if not closed_global.empty:
            g_wins = closed_global[closed_global['Profit'] > 0]
            g_win_rate = (len(g_wins) / len(closed_global)) * 100
            g_size_min = closed_global['Profit'].min() 
            g_size_max = closed_global['Profit'].max()
        else:
            g_win_rate, g_wins, g_size_min, g_size_max = 0, [], 0, 0

        st.markdown("### 📈 Statistiche Operative Portafoglio Totale")
        cg1, cg2, cg3, cg4 = st.columns(4)
        cg1.metric("Global Win Rate", f"{g_win_rate:.1f}%")
        cg2.metric("Total Trades Profittevoli", f"{len(g_wins)} / {len(closed_global)}")
        cg3.metric("Worst Trade Execution", f"€ {g_size_min:,.2f}")
        cg4.metric("Best Trade Execution", f"€ {g_size_max:,.2f}")
        
        if not closed_global.empty:
            daily_pnl = closed_global.groupby('Exit_Date')['Profit'].sum().reset_index()
            daily_pnl['Cum_Profit'] = daily_pnl['Profit'].cumsum()
            colors = ['#00CC96' if p > 0 else '#EF553B' for p in daily_pnl['Profit']]
            
            fig_pnl_g = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pnl_g.add_trace(go.Bar(x=daily_pnl['Exit_Date'], y=daily_pnl['Profit'], marker_color=colors, name='Daily Realized P&L'), secondary_y=False)
            fig_pnl_g.add_trace(go.Scatter(x=daily_pnl['Exit_Date'], y=daily_pnl['Cum_Profit'], name='Cumulative P&L', line=dict(color='#F2C037', width=2)), secondary_y=True)
            fig_pnl_g.update_layout(title="P&L Reale Globale Aggregato", template="plotly_dark", height=300)
            st.plotly_chart(fig_pnl_g, use_container_width=True)
            
        st.markdown("### 📝 Journal Storico Completo (Tutti i Titoli)")
        render_audit_log(all_trades_global)

    # ==========================================
    # TAB 2: DEEP DIVE SINGOLO TITOLO
    # ==========================================
    with tab2:
        valid_tickers = [t for t in config.UNIVERSE if t in data_dict]
        selected_ticker = st.selectbox("Seleziona Titolo per Analisi Isolata", valid_tickers)
        
        if selected_ticker:
            hist_single, events_single, all_trades_single, closed_single = run_simulation(data_dict, [selected_ticker], initial_capital, base_pos, sl_multiplier, vix, breadth)
            
            df_single = data_dict[selected_ticker]
            ticker_prices = df_single['Close']
            tick_eq = (ticker_prices / ticker_prices.iloc[0]) * initial_capital
            bench_eq = (bench / bench.iloc[0]) * initial_capital
            
            res_single = hist_single.join(bench_eq.rename('VWCE')).join(tick_eq.rename('Ticker_BH')).ffill()
            
            c_strat, s_strat, m_strat, n_strat = calc_metrics(res_single['Equity'])
            c_tick, s_tick, m_tick, n_tick = calc_metrics(res_single['Ticker_BH'])
            
            st.markdown(f"### Performance Isolando **{selected_ticker}**")
            colA, colB, colC = st.columns(3)
            colA.metric("CAGR IOVIQUANT", f"{c_strat:.2%}", f"vs B&H Titolo: {c_tick:.2%}")
            colB.metric("Sharpe IOVIQUANT", f"{s_strat:.2f}", f"vs B&H Titolo: {s_tick:.2f}")
            colC.metric("Max Drawdown IOVIQUANT", f"{m_strat:.2%}", f"vs B&H Titolo: {m_tick:.2%}", delta_color="inverse")
            
            # --- AGGIORNAMENTO MATRICE GRAFICA CON 3 EMA E HMM ---
            fig_dd_top = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.07, 
                row_heights=[0.6, 0.4],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]] # Abilita asse asimmetrico per HMM
            )
            
            # 1A: Prezzo e Medie Mobili (Row 1)
            fig_dd_top.add_trace(go.Scatter(x=df_single.index, y=ticker_prices, name='Prezzo', line=dict(color='gray', width=1.5)), row=1, col=1, secondary_y=False)
            fig_dd_top.add_trace(go.Scatter(x=df_single.index, y=df_single['EMA5'], name='EMA 5', line=dict(color=#028450, width=1)), row=1, col=1, secondary_y=False)
            fig_dd_top.add_trace(go.Scatter(x=df_single.index, y=df_single['EMA21'], name='EMA 21', line=dict(color='orange', width=1.2)), row=1, col=1, secondary_y=False)
            fig_dd_top.add_trace(go.Scatter(x=df_single.index, y=df_single['EMA63'], name='EMA 63', line=dict(color='red', width=1.5)), row=1, col=1, secondary_y=False)
            
            # Valore dell'HMM Regolatore (In Grigio di Sfondo su Asse Secondario - Row 1)
            fig_dd_top.add_trace(go.Scatter(x=df_single.index, y=df_single['P_Bull'], name='HMM Regime (Prob Bull)', line=dict(color='rgba(128, 128, 128, 0.4)', width=1.5, dash='dot')), row=1, col=1, secondary_y=True)
            fig_dd_top.update_yaxes(title_text="HMM Probability", secondary_y=True, range=[0, 1.05], row=1, col=1)
            
            if not events_single.empty:
                buys = events_single[events_single['Action'] == 'BUY']
                sells = events_single[events_single['Action'] == 'SELL']
                fig_dd_top.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', name='Buy Marker', marker=dict(symbol='triangle-up', size=12, color='#00CC96')), row=1, col=1, secondary_y=False)
                fig_dd_top.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', name='Sell Marker', marker=dict(symbol='triangle-down', size=12, color='#EF553B')), row=1, col=1, secondary_y=False)
            
            # 1B: Comparazione Equity Curve (Row 2) - B&H del Titolo modificata in Arancione
            fig_dd_top.add_trace(go.Scatter(x=res_single.index, y=res_single['Equity'], name='IOVIQUANT Strategy', line=dict(color='#00CC96')), row=2, col=1)
            fig_dd_top.add_trace(go.Scatter(x=res_single.index, y=res_single['Ticker_BH'], name=f'{selected_ticker} (Buy&Hold)', line=dict(color='#FF8C00', dash='dot')), row=2, col=1)
            fig_dd_top.add_trace(go.Scatter(x=res_single.index, y=res_single['VWCE'], name='VWCE.MI (Buy&Hold)', line=dict(color='#636EFA', dash='dot')), row=2, col=1)
            
            fig_dd_top.update_layout(height=550, title_text="Analisi Tecnica Avanzata ed Equity Curve Scalata", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_dd_top, use_container_width=True)
            
            st.markdown("---")
            
            if not closed_single.empty:
                wins = closed_single[closed_single['Profit'] > 0]
                win_rate = (len(wins) / len(closed_single)) * 100
                size_min = closed_single['Profit'].min() 
                size_max = closed_single['Profit'].max()
            else:
                win_rate, wins, size_min, size_max = 0, [], 0, 0

            st.markdown(f"### 📈 Statistiche Operative & P&L Reale: {selected_ticker}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win Rate", f"{win_rate:.1f}%")
            c2.metric("Trades Profittevoli", f"{len(wins)} / {len(closed_single)}")
            c3.metric("Profitto Minimo", f"€ {size_min:,.2f}")
            c4.metric("Profitto Massimo", f"€ {size_max:,.2f}")
            
            if not closed_single.empty:
                closed_single['Cum_Profit'] = closed_single['Profit'].cumsum()
                colors = ['#00CC96' if p > 0 else '#EF553B' for p in closed_single['Profit']]
                
                fig_dd_pnl = make_subplots(specs=[[{"secondary_y": True}]])
                fig_dd_pnl.add_trace(go.Bar(x=closed_single['Exit_Date'], y=closed_single['Profit'], marker_color=colors, name='Trade Profit'), secondary_y=False)
                fig_dd_pnl.add_trace(go.Scatter(x=closed_single['Exit_Date'], y=closed_single['Cum_Profit'], name='Cum Profit', line=dict(color='#F2C037')), secondary_y=True)
                fig_dd_pnl.update_layout(height=350, title_text="Distribuzione P&L Singolo Asset", template="plotly_dark")
                st.plotly_chart(fig_dd_pnl, use_container_width=True)
            
            st.markdown("### 📝 Journal Storico Deep Dive")
            render_audit_log(all_trades_single)
