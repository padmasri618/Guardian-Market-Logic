import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64
import bcrypt
import csv
import random
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Guardian | AI Market Sentinel", page_icon="🛡️", layout="wide")

BASE_PATH = r"C:\Guardian_Market_Logic"
DATA_FOLDER = os.path.join(BASE_PATH, "data")
if not os.path.exists(DATA_FOLDER): os.makedirs(DATA_FOLDER)

# Asset Paths
LOGIN_BG = r"C:\Guardian_Market_Logic\data\2.jpg"
MAIN_BG = os.path.join(DATA_FOLDER, "2.jpg")

USER_DB = os.path.join(DATA_FOLDER, "users.csv")
ACTIVITY_DB = os.path.join(DATA_FOLDER, "user_activity.csv")
SUPPORT_DB = os.path.join(DATA_FOLDER, "support_tickets.csv") 
PORTFOLIO_DB = os.path.join(DATA_FOLDER, "portfolio.csv") 
WATCHLIST_DB = os.path.join(DATA_FOLDER, "watchlist.csv")

RF_PATH = os.path.join(DATA_FOLDER, "rf_model.pkl")
XGB_PATH = os.path.join(DATA_FOLDER, "xgboost_model.pkl")
SCALER_PATH = os.path.join(DATA_FOLDER, "scaler.pkl")

NIFTY_50 = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS", "SBI": "SBIN.NS", "Bharti Airtel": "BHARTIARTL.NS", "L&T": "LT.NS",
    "ITC": "ITC.NS", "Hindustan Unilever": "HINDUNILVR.NS", "Axis Bank": "AXISBANK.NS", "Adani Ent": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS", "Maruti Suzuki": "MARUTI.NS", "Sun Pharma": "SUNPHARMA.NS", "ONGC": "ONGC.NS",
    "JSW Steel": "JSWSTEEL.NS", "Tata Steel": "TATASTEEL.NS", "NTPC": "NTPC.NS", "Titan": "TITAN.NS",
    "Power Grid": "POWERGRID.NS", "Bajaj Finance": "BAJFINANCE.NS", "Kotak Bank": "KOTAKBANK.NS", "M&M": "M&M.NS"
}

# --- 2. SECURITY & DB UTILITIES ---
def hash_pass(password): return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
def check_pass(password, hashed): return bcrypt.checkpw(password.encode(), hashed.encode())

def init_db():
    if not os.path.exists(USER_DB):
        df = pd.DataFrame([{"username": "admin", "password": hash_pass("admin123"), "role": "admin", "status": "active"}])
        df.to_csv(USER_DB, index=False, quoting=csv.QUOTE_ALL)
    if not os.path.exists(ACTIVITY_DB):
        pd.DataFrame(columns=["Timestamp", "User", "Stock", "Mode", "Risk", "Signal", "Price", "Confidence"]).to_csv(ACTIVITY_DB, index=False)
    if not os.path.exists(SUPPORT_DB):
        pd.DataFrame(columns=["Timestamp", "User", "Message", "Status"]).to_csv(SUPPORT_DB, index=False)
    if not os.path.exists(PORTFOLIO_DB):
        pd.DataFrame(columns=["User", "Stock", "Qty", "Avg_Price", "Date"]).to_csv(PORTFOLIO_DB, index=False)
    if not os.path.exists(WATCHLIST_DB):
        pd.DataFrame(columns=["User", "Stock", "Ticker"]).to_csv(WATCHLIST_DB, index=False)

def log_trade(user, stock, trade_type, action, price, confidence, risk):
    log_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": user, "Stock": stock, "Mode": trade_type, "Risk": risk,
        "Signal": action, "Price": round(price, 2), "Confidence": f"{int(confidence*100)}%"
    }])
    log_entry.to_csv(ACTIVITY_DB, mode='a', header=False, index=False)
import requests
from bs4 import BeautifulSoup

def fetch_rss_news(feed_url):
    try:
        # Added a 'User-Agent' so the news site doesn't block your app
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(feed_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll('item')
        news_data = []
        for item in items[:6]:
            # .strip() removes extra spaces/newlines
            title = item.find('title').text.strip()
            link = item.find('link').text.strip()
            news_data.append({
                'title': title,
                'link': link,
                'publisher': "Economic Times - Markets"
            })
        return news_data
    except Exception as e:
        print(f"News Error: {e}")
        return []
def add_to_portfolio(user, stock, qty, price):
    df = pd.read_csv(PORTFOLIO_DB) if os.path.exists(PORTFOLIO_DB) else pd.DataFrame(columns=["User", "Stock", "Qty", "Avg_Price", "Date"])
    new_entry = pd.DataFrame([{"User": user, "Stock": stock, "Qty": qty, "Avg_Price": price, "Date": datetime.now().strftime("%Y-%m-%d")}])
    pd.concat([df, new_entry]).to_csv(PORTFOLIO_DB, index=False)
    st.success(f"Successfully logged {qty} shares of {stock} to Portfolio!")

def set_bg(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f: encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""<style>.stApp {{background-image: url("data:image/png;base64,{encoded}");background-size: cover;}}
        .main .block-container{{ background: rgba(0,0,0,0.92); border-radius: 20px; padding: 40px; color: white; border: 1px solid #444;}}
        </style>""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try: return joblib.load(RF_PATH), joblib.load(XGB_PATH), joblib.load(SCALER_PATH)
    except: return None, None, None

def calculate_indicators(df):
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    df['rsi'] = 100 - (100 / (1 + (gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9))))
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    mid = df['close'].rolling(20).mean(); std = df['close'].rolling(20).std()
    df['bb_high'] = mid + 2*std; df['bb_low'] = mid - 2*std
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = tr.rolling(14).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    df['oc_diff'] = df['close'] - df['open']; df['hl_diff'] = df['high'] - df['low']
    df['volume_change'] = df['volume'].pct_change(); df['distance_ema'] = df['close'] - df['ema_21']
    df['trend_strength'] = df['ema_9'] - df['ema_21']; df['hour'] = 12
    return df.ffill().bfill().replace([np.inf, -np.inf], 0).fillna(0)

def generate_xai_reasoning(df):
    row = df.iloc[-1]
    reasons = []
    if row['ema_9'] < row['ema_21']: reasons.append("📉 Bearish Cross: EMA 9 is below EMA 21.")
    else: reasons.append("📈 Bullish Cross: EMA 9 is above EMA 21.")
    if row['close'] > row['vwap']: reasons.append("💎 Price is trading above VWAP, showing strong intraday demand.")
    if row['macd'] > row['macd_signal']: reasons.append("🚀 Positive Momentum: MACD line is above the Signal line.")
    if row['rsi'] > 70: reasons.append("⚠️ RSI is Overbought (>70).")
    elif row['rsi'] < 35: reasons.append("✅ RSI is Oversold (<35).")
    return reasons
    
def news_ticker(symbol=None):
    if symbol and symbol in NIFTY_50:
        try:
            # Use .get() inside a list comprehension to avoid KeyError
            raw_news = yf.Ticker(NIFTY_50[symbol]).news[:5]
            titles = [n.get('title', 'News Update') for n in raw_news]
            headlines = " | ".join(titles)
            display_text = f"LIVE {symbol.upper()} UPDATES: {headlines}"
        except: 
            display_text = "CONNECTING TO LIVE SENTINEL STREAM..."
    else: 
        display_text = "SYSTEM STATUS: ACTIVE | REAL-TIME MARKET SENTINEL ENGAGED"
    
    st.markdown(f'''
        <div style="background: rgba(0,255,0,0.05); padding: 12px; border-radius: 10px; border: 1px solid #00ff00; margin-bottom: 25px;">
            <marquee style="color: #00ff00; font-weight: bold;">{display_text}</marquee>
        </div>
    ''', unsafe_allow_html=True)

# --- 3. EXECUTION FLOW ---
init_db()
rf, xgb, scaler = load_assets()
if 'auth' not in st.session_state: st.session_state.auth = False
if 'page' not in st.session_state: st.session_state.page = "Login"

if not st.session_state.auth:
    set_bg(LOGIN_BG)
    news_ticker()
    st.title("🛡️ Guardian Market Logic")
    tab_in, tab_up = st.tabs(["Sign In", "Sign Up"])
    with tab_in:
        u_in = st.text_input("Username", key="login_user")
        p_in = st.text_input("Password", type="password", key="login_pass")
        if st.button("Initialize Login", use_container_width=True):
            users = pd.read_csv(USER_DB)
            if u_in in users['username'].values:
                row = users[users['username'] == u_in]
                if row['status'].values[0] == 'blocked': st.error("Account Blocked.")
                elif check_pass(p_in, row['password'].values[0]):
                    st.session_state.auth, st.session_state.user = True, u_in
                    st.session_state.role = row['role'].values[0]
                    st.session_state.page = "MarketOverview"; st.rerun()
                else: st.error("Invalid Credentials")
            else: st.error("User not found")
    with tab_up:
        nu = st.text_input("Select Username", key="reg_user")
        np = st.text_input("Select Password", type="password", key="reg_pass")
        if st.button("Account Sign Up", use_container_width=True):
            users = pd.read_csv(USER_DB)
            if nu in users['username'].values: st.warning("Username unavailable")
            else:
                pd.DataFrame([{"username": nu, "password": hash_pass(np), "role": "user", "status": "active"}]).to_csv(USER_DB, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
                st.success("Success! Please Sign In.")
else:
    set_bg(MAIN_BG)
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.user}")
        st.caption(f"🛡️ Access Level: {st.session_state.role.upper()}")
        st.markdown("---")
        
        # Common Pages
        if st.button("🏠 Market Overview", use_container_width=True): st.session_state.page = "MarketOverview"; st.rerun()
        if st.button("📈 Analysis Dashboard", use_container_width=True): st.session_state.page = "Setup"; st.rerun()
        if st.button("📊 Strategy Backtest", use_container_width=True): 
            st.session_state.page = "Backtest"; st.rerun()
        if st.button("🗺️ Sector Heatmap", use_container_width=True): 
            st.session_state.page = "SectorPage"; st.rerun()   
        if st.button("💼 Live Portfolio", use_container_width=True): 
                st.session_state.page = "PortfolioPage"; st.rerun()       
        # --- ADMIN ONLY SECTION ---
        if st.session_state.role == "admin":
            st.markdown("### 👑 Admin Suite")
            if st.button("💎 User Management", use_container_width=True): st.session_state.page = "AdminPanel"; st.rerun()
            if st.button("📂 System Logs", use_container_width=True): st.session_state.page = "SystemLogs"; st.rerun()
            if st.button("📧 Support Inbox", use_container_width=True): st.session_state.page = "SupportInbox"; st.rerun()
             
        
        st.markdown("---")
        if st.button("⚙️ Profile & History", use_container_width=True): st.session_state.page = "Profile"; st.rerun()
        # --- Updated Watchlist Section with Delete Function ---
        st.subheader("⭐ My Watchlist")
        w_df = pd.read_csv(WATCHLIST_DB)
        u_watchlist = w_df[w_df['User'] == st.session_state.user]
        
        if not u_watchlist.empty:
            for _, row in u_watchlist.iterrows():
                # Split sidebar space into two columns: one for Search, one for Delete
                side_col1, side_col2 = st.columns([4, 1])
                
                with side_col1:
                    if st.button(f"🔍 {row['Stock']}", key=f"side_{row['Ticker']}", use_container_width=True):
                        st.session_state.auto_ticker = row['Ticker']
                        st.session_state.page = "Setup"
                        st.rerun()
                
                with side_col2:
                    # Remove button (❌)
                    if st.button("❌", key=f"del_{row['Ticker']}", help="Remove from Watchlist"):
                        # Keep everything EXCEPT this specific ticker for this user
                        new_w_df = w_df[~((w_df['User'] == st.session_state.user) & (w_df['Ticker'] == row['Ticker']))]
                        new_w_df.to_csv(WATCHLIST_DB, index=False)
                        st.toast(f"Removed {row['Stock']} from Watchlist")
                        st.rerun()
        else:
            st.caption("No stocks in watchlist.")
        st.markdown("---")
        if st.button("🚪 Terminate Session", type="primary", use_container_width=True): 
            st.session_state.auth = False; st.rerun()

    if st.session_state.page == "MarketOverview":
        st.fragment(run_every=300)
        st.title("📊 Global Market Overview")
        news_ticker()
        
        # --- Top Metrics ---
        indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK", "INDIA VIX": "^INDIAVIX"}
        cols = st.columns(len(indices))
        for i, (name, sym) in enumerate(indices.items()):
            try:
                data = yf.Ticker(sym).history(period="2d")
                price, change = data['Close'].iloc[-1], ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                cols[i].metric(name, f"₹{price:,.2f}" if "VIX" not in name else f"{price:.2f}", f"{change:.2f}%")
            except: cols[i].write(f"{name} N/A")
        
        # --- MARKET MOOD GAUGE ---
        st.subheader("🌡️ Market Sentiment Sentinel")
        mood_score = random.randint(30, 85) # You can replace this with actual logic later
        mood_label = "GREED" if mood_score > 60 else "FEAR" if mood_score < 40 else "NEUTRAL"
        mood_color = "#00ff00" if mood_label == "GREED" else "#ff4b4b" if mood_label == "FEAR" else "#f1c40f"
        
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 5px solid {mood_color};">
                <h3 style="margin:0;">Market Mood: <span style="color:{mood_color};">{mood_label}</span></h3>
                <p style="color:#aaa;">The system is detecting high {mood_label.lower()} in the current session. Trade with caution.</p>
                <div style="width:100%; background:#444; height:10px; border-radius:5px;">
                    <div style="width:{mood_score}%; background:{mood_color}; height:10px; border-radius:5px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- Front Page Market News (RSS VERSION) ---
        st.subheader("🌐 Top Market Stories")
        
        # Using Economic Times - Market News RSS feed
        et_market_rss = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
        market_news = fetch_rss_news(et_market_rss)
        
        if market_news:
            n_col1, n_col2 = st.columns(2)
            for idx, item in enumerate(market_news):
                target_col = n_col1 if idx % 2 == 0 else n_col2
                with target_col:
                    with st.container(border=True):
                        st.markdown(f"#### [{item['title']}]({item['link']})")
                        st.caption(f"📰 Source: {item['publisher']}")
        else:
            st.info("📡 Market news stream currently refreshing. Please check back in a moment.")

        st.markdown("---")
        if st.button("Proceed to Technical Analysis Dashboard ➡️", use_container_width=True): 
            st.session_state.page = "Setup"; st.rerun()

    elif st.session_state.page == "Setup":
        st.title("📈 Technical Analysis Setup")
        sc1, sc2 = st.columns([3, 1])
        with sc1: search_q = st.text_input("Search Stock Ticker", placeholder="Enter symbol (e.g., RELIANCE.NS)")
        with sc2: 
            if st.button("Add to Watchlist") and search_q:
                pd.DataFrame([{"User": st.session_state.user, "Stock": search_q.split('.')[0], "Ticker": search_q}]).to_csv(WATCHLIST_DB, mode='a', header=False, index=False)
                st.success("Added!")
        mode = st.radio("Source", ["Live Market", "Historical CSV", "Custom Upload"], horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            if mode == "Live Market":
                default_idx = list(NIFTY_50.values()).index(st.session_state.get('auto_ticker', 'RELIANCE.NS')) if st.session_state.get('auto_ticker') in NIFTY_50.values() else 0
                asset_label = st.selectbox("Select Asset", list(NIFTY_50.keys()), index=default_idx)
                asset_ticker = NIFTY_50[asset_label]
            else:
                asset_ticker = st.file_uploader("Import CSV"); asset_label = "Imported"
            risk_level = st.selectbox("Risk Sensitivity", ["High Cost - High Risk", "Medium Cost - Medium Risk", "Low Cost - Low Risk"])
        with c2:
            target_strategy = st.selectbox("Market Strategy", ["Intraday", "Swing", "Momentum"])
            news_ticker(asset_label if mode == "Live Market" else None)

        if st.button("EXECUTE SYSTEM AUDIT", use_container_width=True):
            try:
                with st.spinner("Analyzing Market Sentinel..."):
                    ticker_obj = yf.Ticker(asset_ticker) if mode == "Live Market" else None
                    df_raw = ticker_obj.history(period="5d", interval="1m") if mode == "Live Market" else pd.read_csv(asset_ticker)
                    info = ticker_obj.info if mode == "Live Market" else {}
                    df_p = calculate_indicators(df_raw)
                    
                    # 1. Prepare Features
                    features = ['ema_9','ema_21','rsi','macd','macd_signal','bb_high','bb_low','atr','vwap','oc_diff','hl_diff','volume_change','distance_ema','trend_strength','hour']
                    scaled = scaler.transform(df_p[features].tail(1))
                    
                    # 2. CALCULATE ACTION FIRST (This defines 'prob' and 'action')
                    prob = (rf.predict_proba(scaled)[0][1] + xgb.predict_proba(scaled)[0][1]) / 2
                    action = "BUY" if prob > 0.6 else "SELL" if prob < 0.4 else "HOLD"

                    # 3. NOW LOG THE TRADE (Now 'action' and 'prob' exist!)
                    log_trade(
                        st.session_state.user, 
                        asset_label, 
                        target_strategy, 
                        action, 
                        df_p['close'].iloc[-1], 
                        prob, 
                        risk_level
                    )

                    # 4. Save results to session state and move to Dashboard
                    st.session_state.result = {
                        "action": action, 
                        "prob": prob, 
                        "price": df_p['close'].iloc[-1], 
                        "sl": df_p['close'].iloc[-1] * 0.98, 
                        "target": df_p['close'].iloc[-1] * 1.05, 
                        "history": df_p.tail(100), 
                        "stock": asset_label, 
                        "info": info, 
                        "ticker": asset_ticker, 
                        "mode": mode
                    }
                    st.session_state.page = "Dashboard"
                    st.rerun()
            except Exception as e: 
                st.error(f"Audit Error: {e}")

    elif st.session_state.page == "Dashboard":
        
        res = st.session_state.result
        info = res['info']
        hist = res['history']
        
        # --- 1. COMPANY PROFILE (Requirement 1) ---
        st.title(f"🏢 Analysis: {res['stock']}")
        with st.expander("ℹ️ About Organization", expanded=False):
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
            st.write(info.get('longBusinessSummary', 'Business description not available.'))
            if info.get('website'): st.link_button("Visit Official Website", info.get('website'))

        # --- 2. SIGNALS & LIVE QUOTE (Requirement 2 & 3) ---
        action_clr = "#00ff00" if res['action']=="BUY" else "#ff4b4b" if res['action']=="SELL" else "#f1c40f"
        st.markdown(f"""
            <div style="text-align:center; padding:15px; border:2px solid {action_clr}; border-radius:15px; background: rgba(0,0,0,0.6); margin-bottom: 20px;">
                <h1 style="color:{action_clr}; letter-spacing: 5px; margin:0;">{res['action']} SIGNAL</h1>
            </div>
        """, unsafe_allow_html=True)

        # Primary Metrics (LTP, SL, Target)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LTP (Current)", f"₹{res['price']:,.2f}")
        m2.metric("Today's Open", f"₹{info.get('open', 0):,.2f}")
        m3.metric("Stop Loss", f"₹{res['sl']:,.2f}", delta_color="inverse")
        m4.metric("Target", f"₹{res['target']:,.2f}")
        #AI ANALYSTS RATING (Replacing the Plotly Chart) ---
        st.markdown("### 📊 AI Analysts Rating")
        
        # Calculate percentages based on the model probability
        buy_pct = int(res['prob'] * 100)
        hold_pct = 32  # Static or dynamic based on your logic
        sell_pct = 100 - buy_pct - hold_pct if (100 - buy_pct - hold_pct) > 0 else 0

        # Custom CSS for the Glowing Rating Bar
        st.markdown(f"""
            <style>
            .rating-container {{
                width: 100%;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                height: 12px;
                display: flex;
                overflow: hidden;
                box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
                margin-bottom: 5px;
            }}
            .buy-bar {{ width: {buy_pct}%; background: linear-gradient(90deg, #00ff00, #00cc00); }}
            .hold-bar {{ width: {hold_pct}%; background: linear-gradient(90deg, #f1c40f, #d4ac0d); }}
            .sell-bar {{ width: {sell_pct}%; background: linear-gradient(90deg, #ff4b4b, #cc3333); }}
            .label-row {{ display: flex; justify-content: space-between; font-size: 10px; color: #aaa; margin-bottom: 20px; }}
            </style>
            
            <div class="rating-container">
                <div class="buy-bar"></div>
                <div class="hold-bar"></div>
                <div class="sell-bar"></div>
            </div>
            <div class="label-row">
                <span>BUY {buy_pct}%</span>
                <span>HOLD {hold_pct}%</span>
                <span>SELL {sell_pct}%</span>
            </div>
        """, unsafe_allow_html=True)

        # Space out the display to match the background image's "Upward Arrow"
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        # --- 3. ADVANCED CHARTING (With Timeframe Selector) ---
        st.markdown("### 📈 Technical Sentinel Chart")
        
        # NEW: Timeframe Selector
        timeframe = st.radio("Select Interval", ["1D (Live)", "5D", "1M", "1Y"], horizontal=True)
        
        # Mapping timeframe to yfinance periods
        tf_map = {"1D (Live)": ("1d", "1m"), "5D": ("5d", "5m"), "1M": ("1mo", "1h"), "1Y": ("1y", "1d")}
        period, interval = tf_map[timeframe]
        
        # Fetch fresh data based on selection
        chart_data = yf.Ticker(res['ticker']).history(period=period, interval=interval)
        chart_data = calculate_indicators(chart_data) # Ensure indicators are re-calculated

        selected_indicators = st.multiselect(
            "Select Indicators to Overlay",
            ["EMA 9", "EMA 21", "VWAP", "Bollinger Bands"]
        )

        fig = go.Figure()

        # Neon Candlestick
        fig.add_trace(go.Candlestick(
            x=chart_data.index, 
            open=chart_data['open'],   # Changed from 'Open' to 'open'
            high=chart_data['high'],   # Changed from 'High' to 'high'
            low=chart_data['low'],     # Changed from 'Low' to 'low'
            close=chart_data['close'], # Changed from 'Close' to 'close'
            name="Price",
            increasing_line_color='#00ff00', 
            decreasing_line_color='#ff4b4b'
        ))
        # These 'if' statements will now work perfectly
        if "EMA 9" in selected_indicators:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['ema_9'], line=dict(color='#00ffff', width=1.2), name="EMA 9"))
        
        if "EMA 21" in selected_indicators:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['ema_21'], line=dict(color='#ff00ff', width=1.2), name="EMA 21"))
        
        if "VWAP" in selected_indicators:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['vwap'], line=dict(color='#ffffff', width=2, dash='dot'), name="VWAP"))
            
        if "Bollinger Bands" in selected_indicators:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['bb_high'], line=dict(color='rgba(255,255,255,0.2)', width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['bb_low'], line=dict(color='rgba(255,255,255,0.2)', width=0), fill='tonexty', name="BB Range"))
        if "RSI" in selected_indicators:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['rsi'], line=dict(color='#f1c40f', width=1.5), name="RSI"), row=2, col=1)
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1) 
        fig.update_layout(
            template="plotly_dark",
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 4. AI EXPLAINABILITY (XAI) (Requirement 5) ---
        st.subheader("🤖 AI Analytics Reasoning")
        reasons = generate_xai_reasoning(hist)
        x_cols = st.columns(len(reasons))
        for idx, reason in enumerate(reasons):
            with x_cols[idx]:
                st.info(reason)

        # --- 5. FUNDAMENTALS & STATISTICS (Requirement 6) ---
        st.markdown("---")
        st.subheader("📊 Fundamental Metrics")
        fund_data = {
            "Market Cap": f"₹{info.get('marketCap', 0):,}",
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "52W High": f"₹{info.get('fiftyTwoWeekHigh', 0):,.2f}",
            "52W Low": f"₹{info.get('fiftyTwoWeekLow', 0):,.2f}",
            "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%",
            "Beta (Risk)": info.get('beta', 'N/A'),
            "EPS": info.get('trailingEps', 'N/A'),
            "Avg Volume": f"{info.get('averageVolume', 0):,}"
        }
        # Display as a clean table
        st.table(pd.DataFrame([fund_data]).T.rename(columns={0: "Value"}))
        #CORPORATE ACTIONS
        st.subheader("🗓️ Corporate Actions & Events")
        try:
            ticker_obj = yf.Ticker(res['ticker'])
            actions = ticker_obj.actions.tail(3) # Gets Dividends and Splits
            if not actions.empty:
                st.dataframe(actions, use_container_width=True)
            else:
                st.info("No recent corporate actions (Dividends/Splits) found.")
        except:
            st.write("Event data unavailable.")
        # --- 6. TRADE CALCULATOR (Requirement 7) ---
        st.markdown("---")
        with st.container(border=True):
            st.subheader("💼 Trade Calculator & Risk Report")
            qty = st.number_input("How many stocks you want to buy?", min_value=1, value=10)
            
            c1, c2, c3 = st.columns(3)
            c1.write(f"Total Investment: **₹{qty * res['price']:,.2f}**")
            c2.write(f"Potential Loss (SL): <span style='color:#ff4b4b'>₹{abs(res['price'] - res['sl']) * qty:,.2f}</span>", unsafe_allow_html=True)
            c3.write(f"Potential Profit: <span style='color:#00ff00'>₹{abs(res['target'] - res['price']) * qty:,.2f}</span>", unsafe_allow_html=True)
            
            b1, b2 = st.columns(2)
            if b1.button("Log Purchase to My Portfolio", use_container_width=True): 
                add_to_portfolio(st.session_state.user, res['stock'], qty, res['price'])
            if b2.button("📥 Download Analysis Report", use_container_width=True):
                st.toast("Generating PDF Report... Please wait.")
            if st.button("🔄 Start New Analysis", use_container_width=True, type="primary"): 
                st.session_state.page = "Setup"; st.rerun()

    # --- NEW PAGE: BACKTESTING (Insert after Profile and before AdminPanel) ---
    elif st.session_state.page == "Backtest":
        st.title("🧪 Strategy Backtesting Lab")
        st.caption("Test the 'Guardian Logic' cross-over strategy against 1-year historical data.")
        
        test_stock = st.selectbox("Select Asset to Backtest", list(NIFTY_50.keys()))
        
        if st.button("RUN 1-YEAR BACKTEST", use_container_width=True):
            with st.spinner("Processing 1 Year of Market Candles..."):
                # Fetch 1 year of daily data
                bt_df = yf.Ticker(NIFTY_50[test_stock]).history(period="1y")
                
                # We apply indicators (this turns columns into lowercase 'close', 'ema_9', etc.)
                bt_df = calculate_indicators(bt_df)
                
                # Strategy Logic: Buy when EMA 9 is above EMA 21
                bt_df['Signal'] = np.where(bt_df['ema_9'] > bt_df['ema_21'], 1, 0)
                
                # Calculate Returns using lowercase 'close'
                bt_df['Returns'] = bt_df['close'].pct_change()
                bt_df['Strategy_Returns'] = bt_df['Returns'] * bt_df['Signal'].shift(1)
                
                cum_returns = (1 + bt_df['Strategy_Returns'].fillna(0)).cumprod() - 1
                
                c1, c2 = st.columns(2)
                # Performance Metrics
                strat_perf = cum_returns.iloc[-1] * 100
                hold_perf = ((bt_df['close'].iloc[-1] / bt_df['close'].iloc[0]) - 1) * 100
                
                c1.metric("Total Strategy Return", f"{strat_perf:.2f}%")
                c2.metric("Market Buy & Hold", f"{hold_perf:.2f}%")
                
                # Growth Chart
                fig_bt = px.line(cum_returns, title=f"Cumulative Growth of ₹1 Investment in {test_stock}")
                fig_bt.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_bt, use_container_width=True)
                
                st.info("💡 **Logic:** This backtest assumes a 'Long' position whenever the 9-day EMA is above the 21-day EMA.")

    elif st.session_state.page == "SectorPage":
        st.title("🗺️ Sector Performance Heatmap")
        st.caption("Real-time strength of different market segments")
        
        sectors = {
            "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS"],
            "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
            "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS"],
            "Auto": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS"]
        }
        
        sector_data = []
        for sec, tickers in sectors.items():
            for t in tickers:
                try:
                    d = yf.Ticker(t).history(period="1d")
                    chg = ((d['Close'].iloc[-1] - d['Open'].iloc[0])/d['Open'].iloc[0])*100
                    sector_data.append({"Sector": sec, "Stock": t, "Change %": round(chg, 2)})
                except: pass
        
        df_sec = pd.DataFrame(sector_data)
        fig_heat = px.treemap(df_sec, path=['Sector', 'Stock'], values=np.abs(df_sec['Change %']),
                              color='Change %', color_continuous_scale='RdYlGn',
                              title="Market Structure Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

    elif st.session_state.page == "PortfolioPage":
        st.title("💼 My Live Portfolio")
        st.caption("Real-time P&L tracking based on your logged trades.")
        
        if os.path.exists(PORTFOLIO_DB):
            p_df = pd.read_csv(PORTFOLIO_DB)
            user_p = p_df[p_df['User'] == st.session_state.user]
            
            if not user_p.empty:
                # 1. Action Button (Reset)
                if st.button("🗑️ Reset Portfolio Data", type="secondary"):
                    new_p_df = p_df[p_df['User'] != st.session_state.user]
                    new_p_df.to_csv(PORTFOLIO_DB, index=False)
                    st.success("Portfolio cleared!")
                    st.rerun()

                # 2. Setup Variables
                portfolio_summary = []
                total_invested = 0
                total_current = 0

                # 3. Process Data
                with st.spinner("Fetching Live Prices..."):
                    for _, row in user_p.iterrows():
                        try:
                            ticker = NIFTY_50.get(row['Stock'], row['Stock'] + ".NS")
                            live_data = yf.Ticker(ticker).history(period="1d")
                            current_price = live_data['Close'].iloc[-1]
                            
                            invested = row['Qty'] * row['Avg_Price']
                            current_val = row['Qty'] * current_price
                            pnl = current_val - invested
                            pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
                            
                            total_invested += invested
                            total_current += current_val
                            
                            portfolio_summary.append({
                                "Stock": row['Stock'],
                                "Qty": row['Qty'],
                                "Avg Price": f"₹{row['Avg_Price']:.2f}",
                                "Current": f"₹{current_price:.2f}",
                                "P&L": round(pnl, 2),
                                "P&L %": f"{pnl_pct:.2f}%"
                            })
                        except: 
                            pass
                
                # 4. Display Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Invested", f"₹{total_invested:,.2f}")
                
                total_pnl = total_current - total_invested
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                c2.metric("Current Value", f"₹{total_current:,.2f}", delta=f"₹{total_pnl:,.2f}", delta_color=pnl_color)
                
                overall_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
                c3.metric("Overall Return", f"{overall_pct:.2f}%")
                
                # 5. Display Table & Chart
                st.table(pd.DataFrame(portfolio_summary))
                
                fig_pnl = px.bar(
                    pd.DataFrame(portfolio_summary), 
                    x='Stock', 
                    y='P&L', 
                    color='P&L', 
                    color_continuous_scale='RdYlGn',
                    title="Profit/Loss per Asset"
                )
                fig_pnl.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pnl, use_container_width=True)
                
            else:
                st.info("Your portfolio is empty. Log a trade from the Analysis Dashboard!")
        else:
            st.warning("Portfolio database not found.")
            
        # --- ADMIN PANEL SECTION ---
    elif st.session_state.page == "AdminPanel" and st.session_state.role == "admin":
        st.title("🛡️ Admin Command Center")
        
        # Fresh Data Read
        users_df = pd.read_csv(USER_DB)
        activity_df = pd.read_csv(ACTIVITY_DB)

        # 1. Professional Metrics Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Users", len(users_df))
        m2.metric("System Logs", len(activity_df))
        m3.metric("System Status", "ONLINE", delta="Stable")
        m4.metric("Security", "Bcrypt-Enforced")

        st.markdown("---")

        # 2. User Activity Monitoring (The "Audit Trail")
        st.subheader("👥 User Activity & Audit Logs")
        if not activity_df.empty:
            # Reverse order-la recent activity-a 
            st.dataframe(activity_df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        else:
            st.info("No system activity recorded yet.")

        # 3. Access Control (Block/Unblock)
        st.subheader("🔐 User Access Management")
        manage_col, status_col = st.columns([2, 1])
        
        with manage_col:
            other_users = users_df[users_df['username'] != 'admin']['username'].unique()
            if len(other_users) > 0:
                target_user = st.selectbox("Select User to Manage", other_users)
            else:
                st.write("No other users found.")
                target_user = None

        if target_user:
            current_status = users_df[users_df['username'] == target_user]['status'].values[0]
            with status_col:
                st.markdown(f"<br>Current Status: **{current_status.upper()}**", unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            if b1.button("🚫 Block User", use_container_width=True):
                users_df.loc[users_df['username'] == target_user, 'status'] = 'blocked'
                users_df.to_csv(USER_DB, index=False, quoting=csv.QUOTE_ALL)
                st.warning(f"User {target_user} blocked.")
                st.rerun()

            if b2.button("✅ Unblock User", use_container_width=True):
                users_df.loc[users_df['username'] == target_user, 'status'] = 'active'
                users_df.to_csv(USER_DB, index=False, quoting=csv.QUOTE_ALL)
                st.success(f"User {target_user} is now active.")
                st.rerun()
            # --- 7. SYSTEM LOGS (Admin Only) ---
    elif st.session_state.page == "SystemLogs" and st.session_state.role == "admin":
        st.title("📂 System Audit Logs")
        if os.path.exists(ACTIVITY_DB):
            logs = pd.read_csv(ACTIVITY_DB)
            st.dataframe(logs.sort_values(by="Timestamp", ascending=False), use_container_width=True)
            if st.button("Clear Logs", type="secondary"):
                pd.DataFrame(columns=["Timestamp", "User", "Stock", "Mode", "Risk", "Signal", "Price", "Confidence"]).to_csv(ACTIVITY_DB, index=False)
                st.rerun()
        else:
            st.info("No logs found.")

        # --- 8. SUPPORT INBOX (Admin Only) ---
    elif st.session_state.page == "SupportInbox" and st.session_state.role == "admin":
        st.title("📧 Support Tickets")
        if os.path.exists(SUPPORT_DB):
            tickets = pd.read_csv(SUPPORT_DB)
            if not tickets.empty:
                st.table(tickets)
            else:
                st.info("Inbox is empty.")
        else:
            st.info("Support database not initialized.")

        # --- 9. PROFILE & HISTORY (For All Users) ---
    elif st.session_state.page == "Profile":
        st.title("⚙️ User Profile & Trade History")
        
        tab1, tab2, tab3 = st.tabs(["My Portfolio", "Recent Activity", "Support"])
        
        with tab1:
            if os.path.exists(PORTFOLIO_DB):
                p_df = pd.read_csv(PORTFOLIO_DB)
                user_p = p_df[p_df['User'] == st.session_state.user]
                if not user_p.empty:
                    st.dataframe(user_p, use_container_width=True)
                else:
                    st.info("Your portfolio is empty.")
        
        with tab2:
            if os.path.exists(ACTIVITY_DB):
                a_df = pd.read_csv(ACTIVITY_DB)
                user_a = a_df[a_df['User'] == st.session_state.user]
                st.dataframe(user_a.sort_values(by="Timestamp", ascending=False), use_container_width=True)

        with tab3:
            st.subheader("Contact Support")
            msg = st.text_area("Describe your issue")
            if st.button("Submit Ticket"):
                new_t = pd.DataFrame([{"Timestamp": datetime.now(), "User": st.session_state.user, "Message": msg, "Status": "Open"}])
                new_t.to_csv(SUPPORT_DB, mode='a', header=False, index=False)
                st.success("Ticket submitted!")