# ============================================================
# 🌐 IA TRADING PRO – APPLICATION CORRIGÉE & MULTI-TRADES
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="IA Trading Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISÉ
# ============================================================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 1rem;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #1e1e2f;
    padding: 1.5rem;
    border-radius: 1rem;
    text-align: center;
    color: white;
    margin: 0.5rem;
}
.signal-buy {
    background: #4CAF50;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}
.signal-sell {
    background: #f44336;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}
.signal-neutral {
    background: #ff9800;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS DE DONNÉES
# ============================================================
@st.cache_data(ttl=300)
def load_data(ticker, period="5y"):
    """Charge les données via yfinance, puis bascule sur CoinGecko en cas d'échec."""
    # Essai avec yfinance
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            return data
    except Exception as e:
        st.warning(f"yfinance a échoué : {e}. Tentative avec CoinGecko...")

    # Fallback CoinGecko
    try:
        coin_map = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'BNB-USD': 'binancecoin',
            'SOL-USD': 'solana',
            'ADA-USD': 'cardano',
            'XRP-USD': 'ripple'
        }
        coin_id = coin_map.get(ticker, 'bitcoin')
        period_days = {'1y': 365, '2y': 730, '3y': 1095, '5y': 1825}.get(period, 1825)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': period_days, 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        prices = json_data['prices']
        volumes = json_data['total_volumes']
        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        vol_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
        vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
        vol_df.set_index('timestamp', inplace=True)
        df['Volume'] = vol_df['Volume']
        df['Open'] = df['Close'].shift(1)
        df['High'] = df['Close'] * 1.01
        df['Low'] = df['Close'] * 0.99
        df['Open'].fillna(df['Close'], inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if not df.empty:
            st.success("✅ Données récupérées via CoinGecko (source de secours)")
            return df
        else:
            st.error("CoinGecko n'a pas renvoyé de données")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Échec des deux sources de données : {e}")
        return pd.DataFrame()

def calculate_indicators(data):
    """Calcule les indicateurs techniques."""
    df = data.copy()
    close = df['Close']
    for p in [1, 5, 10, 20]:
        df[f'Return_{p}D'] = close.pct_change(p)
    for p in [10, 20, 50, 200]:
        df[f'EMA{p}'] = close.ewm(span=p, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI14'] = 100 - (100 / (1 + gain/loss.replace(0, np.nan)))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    df['VOL_20D'] = df['Return_1D'].rolling(20).std()
    df['VOLUME_MA20'] = df['Volume'].rolling(20).mean()
    df['RELATIVE_VOLUME'] = df['Volume'] / df['VOLUME_MA20']
    df['TREND_SHORT'] = (df['EMA20'] > df['EMA50']).astype(int)
    df['TREND_LONG'] = (df['EMA50'] > df['EMA200']).astype(int)
    df['Target'] = (close.shift(-1) > close).astype(int)
    return df.dropna()

FEATURES = [
    'Return_1D', 'Return_5D', 'Return_10D', 'Return_20D',
    'EMA10', 'EMA20', 'EMA50', 'EMA200',
    'RSI14', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
    'VOL_20D', 'RELATIVE_VOLUME', 'TREND_SHORT', 'TREND_LONG'
]

# ============================================================
# BACKTEST MULTI-POSITIONS
# ============================================================
def run_backtest(data, features, threshold, stop_loss_pct, take_profit_pct,
                 capital_initial, max_positions=5, position_size_pct=20):
    """
    Backtest avec gestion de plusieurs positions simultanées.
    Chaque position utilise `position_size_pct` % de l'équité totale du moment.
    """
    if data.empty:
        return capital_initial, [], [], []
    train_days = min(500, max(50, int(len(data) * 0.7)))
    if len(data) <= train_days + 10:
        st.warning("Pas assez de données pour un backtest fiable. Augmentez la période.")
        return capital_initial, [], [], []

    cash = capital_initial
    positions = []          # liste de dicts: entry_price, shares, stop_loss, take_profit, entry_date
    trades = []             # trades clôturés
    equity_curve = []
    model = None
    scaler = StandardScaler()

    for i in range(train_days, len(data)):
        # Réentraînement périodique
        if model is None or (i - train_days) % 10 == 0:
            train_start = max(0, i - train_days)
            train_data = data.iloc[train_start:i]
            if len(train_data) >= 50:
                X_train = train_data[features]
                y_train = train_data['Target']
                X_train_scaled = scaler.fit_transform(X_train)
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)

        current_price = data['Close'].iloc[i]
        current_date = data.index[i]

        # Prédiction
        if model is None:
            prob_up = 0.5
        else:
            X_test = data[features].iloc[i:i+1]
            X_test_scaled = scaler.transform(X_test)
            try:
                prob_up = model.predict_proba(X_test_scaled)[0][1]
            except:
                prob_up = 0.5

        # --- Gestion des positions ouvertes ---
        to_close = []
        for idx, pos in enumerate(positions):
            exit_price = None
            exit_reason = None
            if current_price <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = 'Stop Loss'
            elif current_price >= pos['take_profit']:
                exit_price = pos['take_profit']
                exit_reason = 'Take Profit'
            elif prob_up < 0.40:   # sortie sur signal faible
                exit_price = current_price
                exit_reason = 'Signal'

            if exit_price is not None:
                # Vente
                proceeds = pos['shares'] * exit_price
                cash += proceeds
                returns = (exit_price / pos['entry_price'] - 1) * 100
                trades.append({
                    'date': current_date,
                    'entry_date': pos['entry_date'],
                    'return': returns,
                    'type': exit_reason,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price
                })
                to_close.append(idx)

        # Fermer les positions (en ordre inverse pour éviter décalage d'index)
        for idx in reversed(to_close):
            positions.pop(idx)

        # --- Entrée de nouvelles positions ---
        trend_ok = data['EMA20'].iloc[i] > data['EMA50'].iloc[i]
        if (prob_up >= threshold and trend_ok and len(positions) < max_positions):
            # Taille de position basée sur l'équité actuelle
            equity_now = cash + sum(p['shares'] * current_price for p in positions)
            invest_amount = equity_now * (position_size_pct / 100)
            if invest_amount > 0 and current_price > 0:
                shares = invest_amount / current_price
                stop_loss = current_price * (1 - stop_loss_pct / 100)
                take_profit = current_price * (1 + take_profit_pct / 100)
                positions.append({
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                cash -= invest_amount

        # Valeur totale du portefeuille
        total_equity = cash + sum(p['shares'] * current_price for p in positions)
        equity_curve.append(total_equity)

    final_equity = equity_curve[-1] if equity_curve else capital_initial
    return final_equity, trades, equity_curve, positions  # positions ouvertes finales (pour info)

# ============================================================
# INTERFACE STREAMLIT
# ============================================================
with st.sidebar:
    st.title("⚙️ Configuration")
    cryptos = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Binance Coin': 'BNB-USD',
        'Solana': 'SOL-USD',
        'Cardano': 'ADA-USD',
        'Ripple': 'XRP-USD'
    }
    crypto_name = st.selectbox("Cryptomonnaie", list(cryptos.keys()))
    ticker = cryptos[crypto_name]

    period = st.select_slider("Historique", options=['1y', '2y', '3y', '5y'], value='5y',
                              help="Choisissez au moins 2 ans pour un backtest fiable")

    st.subheader("🤖 Modèle")
    threshold = st.slider("Seuil de probabilité", 0.50, 0.70, 0.55, 0.01)

    st.subheader("🛡️ Gestion du risque")
    stop_loss = st.slider("Stop Loss (%)", 2.0, 20.0, 5.0, 0.5)
    take_profit = st.slider("Take Profit (%)", 5.0, 50.0, 15.0, 1.0)
    max_positions = st.slider("Nombre max de positions", 1, 5, 3, 1)
    position_size_pct = st.slider("Taille par position (% équité)", 5, 50, 20, 5)

    st.subheader("💰 Capital")
    capital_initial = st.number_input("Capital initial (€)", 100.0, 100000.0, 1000.0, 100.0)

    run_button = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# ============================================================
# PAGE PRINCIPALE
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📈 IA Trading Pro</h1>
    <p>Système de trading multi‑positions avec Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if run_button:
    with st.spinner("🔍 Analyse en cours..."):
        data_raw = load_data(ticker, period)
        if data_raw.empty:
            st.error("Impossible de télécharger les données. Vérifiez votre connexion ou réessayez.")
        else:
            data = calculate_indicators(data_raw)
            if data.empty:
                st.error("Données insuffisantes après calcul des indicateurs. Essayez une période plus longue.")
            elif len(data) < 100:
                st.warning(f"Seulement {len(data)} lignes de données. Les résultats peuvent être peu fiables.")
                capital, trades, equity, open_pos = run_backtest(
                    data, FEATURES, threshold, stop_loss, take_profit,
                    capital_initial, max_positions, position_size_pct
                )
            else:
                capital, trades, equity, open_pos = run_backtest(
                    data, FEATURES, threshold, stop_loss, take_profit,
                    capital_initial, max_positions, position_size_pct
                )

            # Vérifier si le backtest a produit une courbe
            if not equity:
                st.info("Aucun backtest effectué. Vérifiez les données et réessayez.")
            else:
                # ============ RÉSULTATS ============
                st.markdown("## 📊 Résultats")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Capital Final", f"€{capital:,.2f}")
                with col2:
                    return_pct = (capital / capital_initial - 1) * 100
                    st.metric("Performance", f"{return_pct:+.2f}%")
                with col3:
                    if trades:
                        wins = len([t for t in trades if t['return'] > 0])
                        win_rate = wins / len(trades) * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    else:
                        st.metric("Win Rate", "N/A")
                with col4:
                    st.metric("Trades clôturés", len(trades))
                    if open_pos:
                        st.caption(f"Positions ouvertes en fin : {len(open_pos)}")

                # ============ GRAPHIQUE ============
                st.markdown("## 📈 Graphique")
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Performance du capital', 'Drawdown'),
                    vertical_spacing=0.15
                )
                fig.add_trace(
                    go.Scatter(y=equity, mode='lines', name='Capital',
                               line=dict(color='#667eea', width=2)),
                    row=1, col=1
                )
                equity_series = pd.Series(equity)
                drawdown = (equity_series / equity_series.cummax() - 1) * 100
                fig.add_trace(
                    go.Scatter(y=drawdown, mode='lines', name='Drawdown',
                               fill='tozeroy', line=dict(color='#f44336')),
                    row=2, col=1
                )
                fig.update_layout(height=600, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

                # ============ SIGNAL ACTUEL ============
                st.markdown("## 🎯 Signal Actuel")
                latest = data.iloc[-1]
                score = 0
                if latest['EMA20'] > latest['EMA50']: score += 1
                if latest['EMA50'] > latest['EMA200']: score += 1
                if latest['RSI14'] > 50: score += 1
                if latest['MACD'] > latest['MACD_SIGNAL']: score += 1

                if score >= 3:
                    st.markdown('<div class="signal-buy">🟢 ACHAT - Conditions favorables</div>',
                                unsafe_allow_html=True)
                elif score == 2:
                    st.markdown('<div class="signal-neutral">🟡 ATTENTE - Conditions mitigées</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="signal-sell">🔴 ÉVITER - Conditions défavorables</div>',
                                unsafe_allow_html=True)

                # ============ DÉTAILS DES TRADES ============
                if trades:
                    st.markdown("## 📋 Détails des trades clôturés")
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(trades_df, use_container_width=True)
                    csv = trades_df.to_csv(index=False)
                    st.download_button("📥 Télécharger (CSV)", csv,
                                       f"trades_{ticker}.csv", "text/csv")
else:
    st.info("👈 Configurez vos paramètres et cliquez sur 'Lancer l'analyse'")
    st.markdown("""
    ## 📖 Guide d'utilisation
    1. **Choisissez votre cryptomonnaie** dans la barre latérale
    2. **Ajustez les paramètres** (seuil, stop loss, take profit, nombre de positions)
    3. **Cliquez sur 'Lancer l'analyse'** pour démarrer
    4. **Analysez les résultats** et le signal actuel

    ## 🎯 Nouveautés
    - ✅ Gestion multi‑positions (jusqu'à 5 simultanées)
    - ✅ Source de données de secours (CoinGecko)
    - ✅ Plus robuste face aux erreurs de téléchargement
    """)
