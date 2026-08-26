# ============================================================
# 🌐 IA TRADING PRO - APPLICATION PRINCIPALE (CORRIGÉE)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
# FONCTIONS
# ============================================================
@st.cache_data(ttl=300)
def load_data(ticker, period="5y"):
    """Charge les données avec cache"""
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Erreur de téléchargement : {e}")
        return pd.DataFrame()

def calculate_indicators(data):
    """Calcule les indicateurs techniques"""
    df = data.copy()
    close = df['Close']
    
    # Returns
    for p in [1, 5, 10, 20]:
        df[f'Return_{p}D'] = close.pct_change(p)
    
    # EMAs
    for p in [10, 20, 50, 200]:
        df[f'EMA{p}'] = close.ewm(span=p, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI14'] = 100 - (100 / (1 + gain/loss.replace(0, np.nan)))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    
    # Volatilité
    df['VOL_20D'] = df['Return_1D'].rolling(20).std()
    
    # Volume
    df['VOLUME_MA20'] = df['Volume'].rolling(20).mean()
    df['RELATIVE_VOLUME'] = df['Volume'] / df['VOLUME_MA20']
    
    # Tendance
    df['TREND_SHORT'] = (df['EMA20'] > df['EMA50']).astype(int)
    df['TREND_LONG'] = (df['EMA50'] > df['EMA200']).astype(int)
    
    # Cible
    df['Target'] = (close.shift(-1) > close).astype(int)
    
    return df.dropna()

FEATURES = [
    'Return_1D', 'Return_5D', 'Return_10D', 'Return_20D',
    'EMA10', 'EMA20', 'EMA50', 'EMA200',
    'RSI14', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
    'VOL_20D', 'RELATIVE_VOLUME', 'TREND_SHORT', 'TREND_LONG'
]

def run_backtest(data, features, threshold, stop_loss, take_profit, capital_initial):
    """Exécute le backtest avec fenêtre d'entraînement ajustable"""
    if data.empty:
        return capital_initial, [], []
    
    # Déterminer la taille de la fenêtre d'entraînement
    # Au moins 50, au maximum 500, sinon 70% des données
    train_days = min(500, max(50, int(len(data) * 0.7)))
    if len(data) <= train_days + 10:
        st.warning("Pas assez de données pour un backtest fiable. Augmentez la période.")
        return capital_initial, [], []
    
    capital = capital_initial
    position = 0
    entry_price = 0
    stop = 0
    take = 0
    trades = []
    equity = []
    model = None
    scaler = StandardScaler()
    
    for i in range(train_days, len(data)):
        # Réentraînement tous les 10 jours
        if model is None or (i - train_days) % 10 == 0:
            train_start = max(0, i - train_days)
            train_data = data.iloc[train_start:i]
            if len(train_data) < 50:
                continue
            X_train = train_data[features]
            y_train = train_data['Target']
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
        
        # Prédiction
        X_test = data[features].iloc[i:i+1]
        if model is None:
            prob_up = 0.5
        else:
            X_test_scaled = scaler.transform(X_test)
            try:
                prob_up = model.predict_proba(X_test_scaled)[0][1]
            except:
                prob_up = 0.5
        
        current_price = data['Close'].iloc[i]
        
        # Gestion de position
        if position > 0:
            if current_price <= stop:
                returns = (stop / entry_price - 1) * 100
                capital *= (1 + returns/100)
                trades.append({'return': returns, 'type': 'Stop Loss', 'date': data.index[i]})
                position = 0
            elif current_price >= take:
                returns = (take / entry_price - 1) * 100
                capital *= (1 + returns/100)
                trades.append({'return': returns, 'type': 'Take Profit', 'date': data.index[i]})
                position = 0
            elif prob_up < 0.45:
                returns = (current_price / entry_price - 1) * 100
                capital *= (1 + returns/100)
                trades.append({'return': returns, 'type': 'Signal', 'date': data.index[i]})
                position = 0
        
        # Entrée en position
        trend_ok = data['EMA20'].iloc[i] > data['EMA50'].iloc[i]
        
        if position == 0 and prob_up >= threshold and trend_ok:
            position = 1
            entry_price = current_price
            stop = current_price * (1 - stop_loss/100)
            take = current_price * (1 + take_profit/100)
        
        equity.append(capital)
    
    return capital, trades, equity

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Sélection de l'actif
    st.subheader("📊 Actif")
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
    
    # Période
    st.subheader("📅 Période")
    period = st.select_slider(
        "Historique",
        options=['1y', '2y', '3y', '5y'],
        value='5y',
        help="Choisissez au moins 2 ans pour un backtest fiable"
    )
    
    # Paramètres
    st.subheader("🤖 Modèle")
    threshold = st.slider(
        "Seuil de probabilité",
        0.50, 0.70, 0.55, 0.01
    )
    
    st.subheader("🛡️ Gestion du risque")
    stop_loss = st.slider("Stop Loss (%)", 2.0, 20.0, 5.0, 0.5)
    take_profit = st.slider("Take Profit (%)", 5.0, 50.0, 15.0, 1.0)
    
    st.subheader("💰 Capital")
    capital_initial = st.number_input(
        "Capital initial (€)",
        100.0, 100000.0, 1000.0, 100.0
    )
    
    # Bouton
    st.markdown("---")
    run_button = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# ============================================================
# PAGE PRINCIPALE
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📈 IA Trading Pro</h1>
    <p>Système de trading algorithmique avec Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if run_button:
    with st.spinner("🔍 Analyse en cours..."):
        # Chargement des données
        data_raw = load_data(ticker, period)
        if data_raw.empty:
            st.error("Impossible de télécharger les données. Vérifiez votre connexion ou réessayez.")
        else:
            data = calculate_indicators(data_raw)
            
            # Vérification des données
            if data.empty:
                st.error("Données insuffisantes après calcul des indicateurs. Essayez une période plus longue (ex: 5y).")
            elif len(data) < 100:
                st.warning(f"Seulement {len(data)} lignes de données. Les résultats peuvent être peu fiables.")
                capital, trades, equity = run_backtest(data, FEATURES, threshold, stop_loss, take_profit, capital_initial)
            else:
                capital, trades, equity = run_backtest(data, FEATURES, threshold, stop_loss, take_profit, capital_initial)
            
            # ============================================================
            # RÉSULTATS (affichés uniquement si backtest effectué)
            # ============================================================
            if equity:  # Si le backtest a produit des résultats
                st.markdown("## 📊 Résultats")
                
                # Métriques
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
                
                with col4:
                    st.metric("Trades", len(trades))
                
                # Graphique
                st.markdown("## 📈 Graphique")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Performance', 'Drawdown'),
                    vertical_spacing=0.15
                )
                
                # Performance
                fig.add_trace(
                    go.Scatter(
                        y=equity,
                        mode='lines',
                        name='Capital',
                        line=dict(color='#667eea', width=2)
                    ),
                    row=1, col=1
                )
                
                # Drawdown
                equity_series = pd.Series(equity)
                drawdown = (equity_series / equity_series.cummax() - 1) * 100
                fig.add_trace(
                    go.Scatter(
                        y=drawdown,
                        mode='lines',
                        name='Drawdown',
                        fill='tozeroy',
                        line=dict(color='#f44336')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=600,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Signal actuel
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
                
                # Détails des trades
                if trades:
                    st.markdown("## 📋 Détails des trades")
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(trades_df, use_container_width=True)
                    
                    # Téléchargement
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger (CSV)",
                        csv,
                        f"trades_{ticker}.csv",
                        "text/csv"
                    )
            else:
                st.info("Aucun backtest effectué. Vérifiez les données et réessayez.")
else:
    st.info("👈 Configurez vos paramètres et cliquez sur 'Lancer l'analyse'")
    
    # Instructions
    st.markdown("""
    ## 📖 Guide d'utilisation
    
    1. **Choisissez votre cryptomonnaie** dans la barre latérale
    2. **Ajustez les paramètres** selon votre stratégie
    3. **Cliquez sur 'Lancer l'analyse'** pour démarrer
    4. **Analysez les résultats** et le signal actuel
    
    ## 🎯 Fonctionnalités
    
    - ✅ 6 cryptomonnaies supportées
    - ✅ 16 indicateurs techniques
    - ✅ Machine Learning (Random Forest)
    - ✅ Backtesting complet
    - ✅ Graphiques interactifs
    - ✅ Signaux en temps réel
    - ✅ Export des résultats
    
    ## ⚠️ Conseils
    
    - Utilisez une période d'au moins **2 ans** pour un backtest fiable.
    - Ajustez le **seuil de probabilité** pour contrôler la sensibilité.
    - Le **Stop Loss** et le **Take Profit** définissent votre gestion du risque.
    """)
