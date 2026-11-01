import time
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------
# Page config (iPad + desktop friendly)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Flow Scanner | 美股扫描",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------
# Light theme
# ---------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
  --bg:#F6F5F1; --card:#FFFFFF; --line:#E8E5DC; --ink:#15172B; --mute:#6D7190;
  --up:#0F9D6B; --up-ink:#0B7A53; --up-bg:#E6F7EF;
  --down:#E5484D; --down-ink:#C23338; --down-bg:#FDECEC;
  --accent:#4F46E5; --sky:#3B82F6;
}

html{ color-scheme:light; }
html, body, .stApp, [class*="css"]{
  font-family:'Sora','PingFang SC','Noto Sans SC','Microsoft YaHei',sans-serif; color:var(--ink);
}
.stApp{
  background:
    radial-gradient(900px 480px at 8% -8%, rgba(255,190,150,.38), transparent 60%),
    radial-gradient(900px 440px at 100% 0%, rgba(165,180,252,.40), transparent 60%),
    var(--bg);
}
#MainMenu, footer{ visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.2rem; max-width:1250px; }

/* Sidebar */
section[data-testid="stSidebar"]{ background:#FFFFFF; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] h2{ font-size:1.05rem; font-weight:600; color:var(--ink); }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p{ color:var(--ink); }
section[data-testid="stSidebar"] input{
  background:#FAF9F6 !important; border:1px solid var(--line) !important;
  border-radius:10px !important; font-family:'IBM Plex Mono',monospace !important; color:var(--ink) !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button{
  width:100%; border:0; border-radius:12px; padding:.7rem 1rem;
  font-family:'Sora',sans-serif; font-weight:600; color:#fff;
  background:linear-gradient(120deg,var(--accent),var(--sky));
  box-shadow:0 8px 20px -6px rgba(79,70,229,.55);
  transition:transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform:translateY(-1px); box-shadow:0 12px 26px -6px rgba(79,70,229,.65); color:#fff;
}
.stDownloadButton > button{
  width:auto; color:var(--ink); background:var(--card); border:1px solid var(--line);
  box-shadow:0 1px 2px rgba(20,22,43,.05);
}
.stDownloadButton > button:hover{ color:var(--accent); border-color:var(--accent); box-shadow:0 6px 16px -8px rgba(79,70,229,.5); }

/* Hero */
.hero{ margin-bottom:1.4rem; }
.hero-title{
  font-size:2.55rem; font-weight:700; letter-spacing:-.02em; line-height:1.1;
  background:linear-gradient(100deg,var(--ink) 25%,var(--accent) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero-sub{ color:var(--mute); margin-top:.55rem; font-size:1rem; max-width:62ch; }

/* Stat strip */
.stats{ display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:0 0 1.6rem; }
.stat{
  background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1rem 1.15rem;
  box-shadow:0 1px 2px rgba(20,22,43,.04), 0 14px 30px -18px rgba(79,70,229,.25);
}
.stat-v{ font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:1.55rem; }
.stat-l{ color:var(--mute); font-size:.82rem; margin-top:.25rem; }
@media (max-width:760px){ .stats{ grid-template-columns:repeat(2,1fr); } .hero-title{ font-size:1.9rem; } }

/* Signal panels */
.panel{
  background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:1.15rem 1.25rem 1.3rem; margin-bottom:1.1rem; position:relative; overflow:hidden;
  box-shadow:0 1px 2px rgba(20,22,43,.04), 0 18px 36px -22px rgba(20,22,43,.25);
}
.panel::before{ content:""; position:absolute; inset:0 0 auto 0; height:3px; }
.panel.up::before{ background:linear-gradient(90deg,var(--up),rgba(15,157,107,0)); }
.panel.down::before{ background:linear-gradient(90deg,var(--down),rgba(229,72,77,0)); }
.panel-head{ display:flex; align-items:center; gap:.8rem; margin-bottom:.95rem; }
.panel-title{ font-weight:600; font-size:1.02rem; }
.panel-sub{ color:var(--mute); font-size:.8rem; margin-top:.15rem; }
.dot{ width:10px; height:10px; border-radius:50%; flex:none; }
.dot.up{ background:var(--up); box-shadow:0 0 0 4px var(--up-bg); }
.dot.down{ background:var(--down); box-shadow:0 0 0 4px var(--down-bg); }
.count{
  margin-left:auto; font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:1.2rem;
  padding:.1rem .7rem; border-radius:10px;
}
.panel.up .count{ background:var(--up-bg); color:var(--up-ink); }
.panel.down .count{ background:var(--down-bg); color:var(--down-ink); }

.chips{ display:flex; flex-wrap:wrap; gap:.5rem; }
.chip{
  font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:.88rem;
  padding:.38rem .8rem; border-radius:999px; text-decoration:none !important;
  transition:transform .12s ease, box-shadow .12s ease;
}
.chip .v{ font-weight:500; opacity:.7; margin-left:.45rem; font-size:.78rem; }
.chip.up{ color:var(--up-ink); background:var(--up-bg); border:1px solid rgba(15,157,107,.28); }
.chip.down{ color:var(--down-ink); background:var(--down-bg); border:1px solid rgba(229,72,77,.28); }
.chip:hover{ transform:translateY(-2px); box-shadow:0 8px 16px -8px rgba(20,22,43,.35); }
.empty{ color:var(--mute); font-size:.88rem; }

/* Notices */
.notice{
  background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent);
  border-radius:12px; padding:.9rem 1.1rem; color:var(--mute); margin-bottom:1.2rem;
  box-shadow:0 1px 2px rgba(20,22,43,.04);
}
.notice b{ color:var(--ink); font-weight:600; }
.notice.ok{ border-left-color:var(--up); }
.notice.warn{ border-left-color:#F59E0B; }

/* Progress bar */
.stProgress > div > div > div > div{ background:linear-gradient(90deg,var(--accent),var(--sky)); }
.stProgress p{ font-family:'IBM Plex Mono',monospace; color:var(--mute); font-size:.85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Language (English / 中文)
# ---------------------------------------------------------------
T = {
    "en": {
        "hero_title": "US Stock Flow Scanner",
        "hero_sub": "Institutional flow and trend signals across the S&P 500, Nasdaq 100 and hot liquid stocks. "
                    "Tap any ticker to open it on Yahoo Finance.",
        "settings": "⚙️ Scan Settings",
        "universe": "Select Stock Universe",
        "modes": ["Auto scan S&P 500, Nasdaq 100 & Hot Liquid Stocks", "Custom stock tickers"],
        "custom": "Enter tickers (comma separated, e.g., AAPL, TSLA, NVDA)",
        "spinner": "Loading S&P 500, Nasdaq 100 & Hot Liquid Stocks Universe...",
        "fallback": "Couldn't load the live index lists, so a backup list of <b>{n}</b> large caps is being used.",
        "tf_label": "Scan Timeframe",
        "tf": {"Daily": "Daily", "Weekly": "Weekly"},
        "run": "Run Scanner",
        "s_universe": "Stocks in universe", "s_tf": "Timeframe", "s_found": "Signals found", "s_time": "Scan time",
        "scanning": "Scanning <b>{n}</b> stocks on the <b>{tf}</b> timeframe. This can take a minute.",
        "start": "Starting scan",
        "progress": "Scanning {t} ({i} of {n})",
        "done": "Scan complete. Checked <b>{n}</b> stocks.",
        "none": "No signals matched current criteria.",
        "keias_buy": "KEIAS Buy Signals", "keias_buy_sub": "Score crossed above 60",
        "kxkx_buy": "KXKX Institutional Accumulation", "kxkx_buy_sub": "Heavy buying near the lows",
        "keias_sell": "KEIAS Sell Signals", "keias_sell_sub": "Score crossed below -60",
        "kxkx_sell": "KXKX Institutional Distribution", "kxkx_sell_sub": "Heavy selling near the highs",
        "empty": "No matches in this scan",
        "table": "View as table",
        "c_ticker": "Ticker", "c_signal": "Signal", "c_value": "KEIAS Value",
        "download": "Download signals (CSV)",
        "idle": "Choose a universe and timeframe in the sidebar, then press <b>Run Scanner</b>. "
                "Results show up here as tappable ticker chips.",
    },
    "zh": {
        "hero_title": "美股资金流向扫描器",
        "hero_sub": "在标普500、纳斯达克100及热门高流动性股票中扫描机构资金流向与趋势信号。点击任意代码即可在 Yahoo Finance 打开。",
        "settings": "⚙️ 扫描设置",
        "universe": "选择股票范围",
        "modes": ["自动扫描标普500、纳斯达克100及热门高流动性股票", "自定义股票代码"],
        "custom": "输入股票代码（逗号分隔，例如 AAPL, TSLA, NVDA）",
        "spinner": "正在加载标普500、纳斯达克100及热门高流动性股票…",
        "fallback": "无法加载实时指数成分股，当前使用 <b>{n}</b> 只大盘股的备用名单。",
        "tf_label": "扫描周期",
        "tf": {"Daily": "日线", "Weekly": "周线"},
        "run": "开始扫描",
        "s_universe": "股票池数量", "s_tf": "周期", "s_found": "信号数量", "s_time": "扫描用时",
        "scanning": "正在扫描 <b>{n}</b> 只股票，周期：<b>{tf}</b>。可能需要一分钟。",
        "start": "开始扫描",
        "progress": "正在扫描 {t}（{i}/{n}）",
        "done": "扫描完成，共检查 <b>{n}</b> 只股票。",
        "none": "没有股票符合当前条件。",
        "keias_buy": "KEIAS 买入信号", "keias_buy_sub": "评分上穿 60",
        "kxkx_buy": "KXKX 机构吸筹", "kxkx_buy_sub": "低位放量买入",
        "keias_sell": "KEIAS 卖出信号", "keias_sell_sub": "评分下穿 -60",
        "kxkx_sell": "KXKX 机构派发", "kxkx_sell_sub": "高位放量卖出",
        "empty": "本次扫描没有符合条件的股票",
        "table": "表格视图",
        "c_ticker": "代码", "c_signal": "信号", "c_value": "KEIAS 数值",
        "download": "下载信号 (CSV)",
        "idle": "在侧边栏选择股票范围和周期，然后点击 <b>开始扫描</b>。结果会以可点击的代码标签显示在这里。",
    },
}

lang = st.sidebar.radio("Language / 语言", ["English", "中文"], horizontal=True)
L = T["zh" if lang == "中文" else "en"]


# ---------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------
def stat(label, value):
    return f'<div class="stat"><div class="stat-v">{value}</div><div class="stat-l">{label}</div></div>'


def panel(title, subtitle, items, kind):
    """items = list of (ticker, value_or_None)."""
    if items:
        chips = "".join(
            f'<a class="chip {kind}" href="https://finance.yahoo.com/quote/{t}" target="_blank">{t}'
            + (f'<span class="v">{v:.1f}</span>' if v is not None else "")
            + "</a>"
            for t, v in items
        )
    else:
        chips = f'<span class="empty">{L["empty"]}</span>'
    return (
        f'<div class="panel {kind}"><div class="panel-head"><span class="dot {kind}"></span>'
        f'<div><div class="panel-title">{title}</div><div class="panel-sub">{subtitle}</div></div>'
        f'<div class="count">{len(items)}</div></div><div class="chips">{chips}</div></div>'
    )


st.markdown(
    f'<div class="hero"><div class="hero-title">{L["hero_title"]}</div>'
    f'<div class="hero-sub">{L["hero_sub"]}</div></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.header(L["settings"])
scan_mode = st.sidebar.radio(L["universe"], L["modes"])


# Fetch the comprehensive pool for Auto Scan
@st.cache_data(ttl=86400)
def get_comprehensive_market_universe():
    ua = {"User-Agent": "Mozilla/5.0 (compatible; FlowScanner/1.0)"}

    def read_tables(url):
        r = requests.get(url, headers=ua, timeout=15)
        r.raise_for_status()
        return pd.read_html(StringIO(r.text))

    extra_hot_tickers = [
        "PLTR", "COIN", "MARA", "RIOT", "SQ", "SHOP", "ROKU", "U", "ARM", "SMCI",
        "MSTR", "HOOD", "SNOW", "DDOG", "NET", "CRWD", "ZS", "PANW", "TEAM", "MELI",
        "SE", "BABA", "JD", "PDD", "NIO", "XPEV", "LI", "TSM", "ASML", "MU", "QCOM",
        "PYPL", "ABNB", "UBER", "LYFT", "DASH", "PINS", "SNAP", "RBLX", "DKNG", "PATH",
        "CELH", "ELF", "RDDT", "ASTS", "LUNR", "RKLB", "IONQ", "RGTI", "CVNA",
    ]
    try:
        # 1. S&P 500
        table_sp = read_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp_tickers = table_sp[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()

        # 2. Nasdaq 100 (optional: S&P alone is still a good universe)
        ndx_tickers = []
        try:
            table_ndx = read_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
            df_ndx = next((t for t in table_ndx if "Ticker" in t.columns or "Symbol" in t.columns), None)
            if df_ndx is not None:
                col = "Ticker" if "Ticker" in df_ndx.columns else "Symbol"
                ndx_tickers = df_ndx[col].astype(str).str.replace(".", "-", regex=False).tolist()
        except Exception:
            pass

        # 3. Hot, liquid growth / tech stocks
        return sorted(set(sp_tickers + ndx_tickers + extra_hot_tickers)), True
    except Exception:
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
            "AVGO", "COST", "PEP", "TMUS", "CSCO", "CRM", "QCOM", "TXN", "AMAT", "IBM",
            "GE", "CAT", "AXP", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "UNH", "JNJ",
            "LLY", "XOM", "CVX", "COP", "SLB", "PFE", "ABBV", "MRK", "T", "VZ", "WMT", "HD",
            "PLTR", "COIN", "MARA", "SQ", "SHOP", "ROKU", "ARM", "SMCI", "MSTR", "HOOD",
        ], False


universe_live = True
if scan_mode == L["modes"][1]:
    custom_tickers = st.sidebar.text_input(
        L["custom"],
        "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, NFLX, INTC",
    )
    tickers = [s.strip().upper() for s in custom_tickers.split(",") if s.strip()]
else:
    with st.spinner(L["spinner"]):
        tickers, universe_live = get_comprehensive_market_universe()

timeframe = st.sidebar.selectbox(L["tf_label"], ["Daily", "Weekly"], format_func=lambda x: L["tf"][x])
interval_val = "1d" if timeframe == "Daily" else "1wk"


# ---------------------------------------------------------------
# Indicator logic
# ---------------------------------------------------------------
def process_single_ticker(ticker, interval_val):
    try:
        # Weekly needs more history (1y of weekly bars is only ~52, below the 60-bar minimum)
        period = "1y" if interval_val == "1d" else "5y"
        # Ticker.history is thread-safe; yf.download can mix up data between threads
        df = yf.Ticker(ticker).history(period=period, interval=interval_val)
        if df.empty or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"]

        # --- KEIAS Indicator Logic ---
        win = 252 if interval_val == "1d" else 52  # 52 weeks either way
        ll52 = low.rolling(win).min() if len(df) >= win else low.min()
        hh52 = high.rolling(win).max() if len(df) >= win else high.max()
        pos_score = np.where(hh52 == ll52, 0, (close - ll52) / (hh52 - ll52) * 200 - 100)

        ma50 = close.rolling(50).mean()
        trend_score = np.where(close > ma50, 100, -100)

        coef = np.where(high == low, 0.001, (2 * close - high - low) / (high - low))
        raw_ev = vol * coef
        sd_ev = raw_ev.rolling(30).std()
        vol_score = np.where(sd_ev == 0, 0, (raw_ev - raw_ev.rolling(30).mean()) / sd_ev * 100)
        vol_score_lim = np.clip(vol_score, -100, 100)

        raw_ias = pos_score * 0.3 + trend_score * 0.4 + vol_score_lim * 0.3
        ias = pd.Series(raw_ias, index=df.index).rolling(3).mean()

        current_ias = float(ias.iloc[-1]) if len(ias) > 0 else 0.0
        prev_ias = float(ias.iloc[-2]) if len(ias) > 1 else 0.0

        keias_buy = bool(current_ias > 60 and prev_ias <= 60)
        keias_sell = bool(current_ias < -60 and prev_ias >= -60)

        # --- KXKX Indicator Logic ---
        vma20 = vol.rolling(20).mean()
        vma60 = vol.rolling(60).mean()
        amp = np.where((high - low) == 0, 0.001, high - low)
        pos_kx = ((close - low) - (high - close)) / amp
        net_inst = pos_kx * vol

        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        safe_zone = (close > ema50) | (ema50 > ema200)

        zqxsl = (vol.rolling(30).min() < vma60 * 0.5)
        zcxdw = (close <= close.rolling(60).min() * 1.15)
        szjbf = (vol > vma20 * 2.0) & (net_inst > 0)

        inst_buy_signal = safe_zone & zqxsl & zcxdw & szjbf
        kxkx_buy = bool(len(inst_buy_signal) > 0 and inst_buy_signal.iloc[-1])

        zcxgw = (close >= high.rolling(60).max() * 0.95)
        inst_sell_signal = zcxgw & (vol > vma20 * 1.8) & (net_inst < 0)
        kxkx_sell = bool(len(inst_sell_signal) > 0 and inst_sell_signal.iloc[-1])

        return {
            "ticker": ticker,
            "keias_val": round(current_ias, 2),
            "keias_buy": keias_buy,
            "keias_sell": keias_sell,
            "kxkx_buy": kxkx_buy,
            "kxkx_sell": kxkx_sell,
        }
    except Exception:
        return None


# ---------------------------------------------------------------
# Stat strip (updates after the scan)
# ---------------------------------------------------------------
stats_slot = st.empty()


def render_stats(found="-", elapsed="-"):
    stats_slot.markdown(
        '<div class="stats">'
        + stat(L["s_universe"], len(tickers))
        + stat(L["s_tf"], L["tf"][timeframe])
        + stat(L["s_found"], found)
        + stat(L["s_time"], elapsed)
        + "</div>",
        unsafe_allow_html=True,
    )


render_stats()

if not universe_live:
    st.markdown(f'<div class="notice warn">{L["fallback"].format(n=len(tickers))}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if st.sidebar.button(L["run"]):
    t0 = time.time()
    st.markdown(
        f'<div class="notice">{L["scanning"].format(n=len(tickers), tf=L["tf"][timeframe].lower())}</div>',
        unsafe_allow_html=True,
    )

    results_data = []
    progress_bar = st.progress(0, text=L["start"])
    completed_count = 0

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_single_ticker, t, interval_val): t for t in tickers}

        for future in as_completed(futures):
            completed_count += 1
            progress_bar.progress(
                completed_count / len(tickers),
                text=L["progress"].format(t=futures[future], i=completed_count, n=len(tickers)),
            )

            res = future.result()
            if res and (res["keias_buy"] or res["keias_sell"] or res["kxkx_buy"] or res["kxkx_sell"]):
                results_data.append(res)

    progress_bar.empty()
    elapsed = time.time() - t0

    keias_b = sorted([r for r in results_data if r["keias_buy"]], key=lambda r: -r["keias_val"])
    keias_s = sorted([r for r in results_data if r["keias_sell"]], key=lambda r: r["keias_val"])
    kxkx_b = sorted([r for r in results_data if r["kxkx_buy"]], key=lambda r: r["ticker"])
    kxkx_s = sorted([r for r in results_data if r["kxkx_sell"]], key=lambda r: r["ticker"])

    total_signals = len(keias_b) + len(keias_s) + len(kxkx_b) + len(kxkx_s)
    render_stats(total_signals, f"{elapsed:.0f}s")

    if not results_data:
        st.markdown(f'<div class="notice warn">{L["none"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="notice ok">{L["done"].format(n=len(tickers))}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                panel(L["keias_buy"], L["keias_buy_sub"], [(r["ticker"], r["keias_val"]) for r in keias_b], "up")
                + panel(L["kxkx_buy"], L["kxkx_buy_sub"], [(r["ticker"], None) for r in kxkx_b], "up"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                panel(L["keias_sell"], L["keias_sell_sub"], [(r["ticker"], r["keias_val"]) for r in keias_s], "down")
                + panel(L["kxkx_sell"], L["kxkx_sell_sub"], [(r["ticker"], None) for r in kxkx_s], "down"),
                unsafe_allow_html=True,
            )

        rows = (
            [(r["ticker"], L["keias_buy"], r["keias_val"]) for r in keias_b]
            + [(r["ticker"], L["keias_sell"], r["keias_val"]) for r in keias_s]
            + [(r["ticker"], L["kxkx_buy"], None) for r in kxkx_b]
            + [(r["ticker"], L["kxkx_sell"], None) for r in kxkx_s]
        )
        df_out = pd.DataFrame(rows, columns=[L["c_ticker"], L["c_signal"], L["c_value"]])

        with st.expander(L["table"]):
            st.dataframe(df_out, hide_index=True)

        st.download_button(
            L["download"],
            df_out.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"signals_{timeframe.lower()}.csv",
            mime="text/csv",
        )
else:
    st.markdown(f'<div class="notice">{L["idle"]}</div>', unsafe_allow_html=True)
