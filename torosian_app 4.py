"""
╔══════════════════════════════════════════════════════════════╗
║       TOROSIAN STOCK INSIGHTS  —  v3                         ║
║  RUN:  python -m streamlit run torosian_app.py               ║
║  DEPS: pip install streamlit yfinance pandas numpy plotly    ║
║             pandas_ta scipy                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# optional deps — imported lazily so app still loads if missing
try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Torosian Stock Insights", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;1,9..144,300&family=DM+Mono:wght@300;400;500&display=swap');
:root{--ink:#0b0a09;--surface:#111210;--surface2:#181917;--surface3:#1f2020;
      --border:#2a2c2b;--gold:#c8953a;--gold-light:#e8c97a;--cream:#e8e4dc;
      --muted:#6b6e6c;--green:#4caf7d;--red:#e05c5c;--cyan:#5bc8c8;}
html,body,[data-testid="stAppViewContainer"]{background:var(--surface)!important;color:var(--cream)!important;font-family:'DM Mono',monospace!important;}
[data-testid="stSidebar"]{background:var(--ink)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{font-family:'DM Mono',monospace!important;color:var(--cream)!important;}
h1,h2,h3{font-family:'Fraunces',serif!important;font-weight:300!important;}
[data-testid="stTabs"] button{font-family:'DM Mono',monospace!important;font-size:11px!important;
  letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--muted)!important;
  background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;padding:8px 18px!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--gold)!important;border-bottom:2px solid var(--gold)!important;}
[data-testid="stTabsContent"]{border:none!important;padding-top:20px!important;}
[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div{background:var(--surface3)!important;
  border:1px solid #353836!important;border-radius:3px!important;color:var(--cream)!important;
  font-family:'DM Mono',monospace!important;font-size:12px!important;}
[data-testid="stButton"]>button{background:var(--gold)!important;color:var(--ink)!important;border:none!important;
  font-family:'DM Mono',monospace!important;font-size:11px!important;font-weight:500!important;
  letter-spacing:.1em!important;text-transform:uppercase!important;border-radius:2px!important;padding:10px 24px!important;}
[data-testid="stButton"]>button:hover{background:var(--gold-light)!important;}
[data-testid="stExpander"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:3px!important;}
[data-testid="stNumberInput"]>div>div>input{background:var(--surface3)!important;border:1px solid #353836!important;
  color:var(--cream)!important;font-family:'DM Mono',monospace!important;}
[data-testid="stDateInput"]>div>div>input{background:var(--surface3)!important;border:1px solid #353836!important;
  color:var(--cream)!important;font-family:'DM Mono',monospace!important;}
hr{border-color:var(--border)!important;}
#MainMenu,footer,header{visibility:hidden!important;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "learn"

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
# ── GICS sector → our 7-sector mapping ───────────────────
GICS_MAP = {
    "Information Technology":  "Technology",
    "Communication Services":  "Technology",
    "Health Care":             "Healthcare",
    "Financials":              "Finance",
    "Consumer Discretionary":  "Consumer",
    "Consumer Staples":        "Consumer",
    "Energy":                  "Energy",
    "Real Estate":             "Real Estate",
    "Industrials":             "Industrials",
    "Materials":               "Industrials",
    "Utilities":               "Energy",
}

# ── Hardcoded fallback (used if Wikipedia fetch fails) ────
FALLBACK_UNIVERSE = [
    ("AAPL","Apple","Technology","Large"),("MSFT","Microsoft","Technology","Large"),
    ("NVDA","Nvidia","Technology","Large"),("GOOGL","Alphabet","Technology","Large"),
    ("META","Meta","Technology","Large"),("INTC","Intel","Technology","Large"),
    ("CSCO","Cisco","Technology","Large"),("IBM","IBM","Technology","Large"),
    ("ORCL","Oracle","Technology","Large"),("ADBE","Adobe","Technology","Large"),
    ("PLTR","Palantir","Technology","Mid"),("DDOG","Datadog","Technology","Mid"),
    ("CRWD","CrowdStrike","Technology","Large"),("SMCI","Super Micro","Technology","Mid"),
    ("JNJ","Johnson & Johnson","Healthcare","Large"),("UNH","UnitedHealth","Healthcare","Large"),
    ("PFE","Pfizer","Healthcare","Large"),("ABBV","AbbVie","Healthcare","Large"),
    ("MRK","Merck","Healthcare","Large"),("LLY","Eli Lilly","Healthcare","Large"),
    ("TMO","Thermo Fisher","Healthcare","Large"),
    ("JPM","JPMorgan Chase","Finance","Large"),("BAC","Bank of America","Finance","Large"),
    ("GS","Goldman Sachs","Finance","Large"),("MS","Morgan Stanley","Finance","Large"),
    ("V","Visa","Finance","Large"),("BRK-B","Berkshire Hathaway","Finance","Large"),
    ("AXP","American Express","Finance","Large"),("WFC","Wells Fargo","Finance","Large"),
    ("AMZN","Amazon","Consumer","Large"),("TSLA","Tesla","Consumer","Large"),
    ("WMT","Walmart","Consumer","Large"),("MCD","McDonald's","Consumer","Large"),
    ("NKE","Nike","Consumer","Large"),("SBUX","Starbucks","Consumer","Large"),
    ("HD","Home Depot","Consumer","Large"),("TGT","Target","Consumer","Large"),
    ("XOM","ExxonMobil","Energy","Large"),("CVX","Chevron","Energy","Large"),
    ("COP","ConocoPhillips","Energy","Large"),("NEE","NextEra Energy","Energy","Large"),
    ("SLB","Schlumberger","Energy","Large"),
    ("AMT","American Tower","Real Estate","Large"),("PLD","Prologis","Real Estate","Large"),
    ("EQIX","Equinix","Real Estate","Large"),("SPG","Simon Property","Real Estate","Large"),
    ("BA","Boeing","Industrials","Large"),("CAT","Caterpillar","Industrials","Large"),
    ("GE","GE Aerospace","Industrials","Large"),("HON","Honeywell","Industrials","Large"),
    ("UPS","UPS","Industrials","Large"),("MMM","3M","Industrials","Large"),
]

@st.cache_data(ttl=86400, show_spinner=False)   # refresh once per day
def load_sp500_universe():
    """
    Fetches the live S&P 500 constituent list from Wikipedia.
    Maps GICS sectors to our 7-sector model.
    All S&P 500 members qualify as Large cap (min ~$14B market cap).
    Falls back to FALLBACK_UNIVERSE if the fetch or parse fails.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        df = tables[0]

        # Normalise column names (Wikipedia occasionally renames them)
        df.columns = [c.strip() for c in df.columns]
        symbol_col   = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
        name_col     = next((c for c in df.columns if "security" in c.lower() or "company" in c.lower() or "name" in c.lower()), None)
        sector_col   = next((c for c in df.columns if "gics sector" in c.lower() or "sector" in c.lower()), None)

        if not all([symbol_col, name_col, sector_col]):
            return FALLBACK_UNIVERSE, True   # True = used fallback

        universe = []
        for _, row in df.iterrows():
            ticker  = str(row[symbol_col]).strip().replace(".", "-")  # BRK.B → BRK-B
            name    = str(row[name_col]).strip()
            gics    = str(row[sector_col]).strip()
            sector  = GICS_MAP.get(gics, "Industrials")
            universe.append((ticker, name, sector, "Large"))

        return universe, False   # False = used live data

    except Exception:
        return FALLBACK_UNIVERSE, True

# Initialise universe — loaded once, cached for 24 hours
if "universe_loaded" not in st.session_state:
    _u, _fallback = load_sp500_universe()
    st.session_state.stock_universe   = _u
    st.session_state.universe_fallback= _fallback
    st.session_state.universe_loaded  = True

STOCK_UNIVERSE = st.session_state.stock_universe
RISK_RANGES={"Low (β < 0.8)":(0.0,0.8),"Medium (β 0.8–1.3)":(0.8,1.3),"High (β > 1.3)":(1.3,99.0)}
MA_NAMES=["EMA 10","SMA 10","EMA 20","SMA 20","EMA 30","SMA 30",
          "EMA 50","SMA 50","EMA 100","SMA 100","EMA 200","SMA 200","Ichimoku","VWMA 20","HMA 9"]
OSC_NAMES=["RSI 14","Stoch %K","CCI 20","ADX 14","Awe. Osc.",
           "Momentum","MACD","Stoch RSI","Williams %R","Bull/Bear","Ult. Osc."]
ALL_INDICATORS=MA_NAMES+OSC_NAMES
SECTOR_THRESHOLDS={
    "Technology":  {"rsi_os":35,"rsi_ob":73,"cci_os":-110,"cci_ob":110,"wpr_os":-82,"wpr_ob":-18},
    "Healthcare":  {"rsi_os":30,"rsi_ob":68,"cci_os":-100,"cci_ob":100,"wpr_os":-80,"wpr_ob":-20},
    "Finance":     {"rsi_os":30,"rsi_ob":68,"cci_os":-100,"cci_ob":105,"wpr_os":-80,"wpr_ob":-20},
    "Consumer":    {"rsi_os":32,"rsi_ob":70,"cci_os":-105,"cci_ob":105,"wpr_os":-80,"wpr_ob":-20},
    "Energy":      {"rsi_os":28,"rsi_ob":72,"cci_os":-115,"cci_ob":115,"wpr_os":-82,"wpr_ob":-18},
    "Real Estate": {"rsi_os":30,"rsi_ob":65,"cci_os": -95,"cci_ob": 95,"wpr_os":-78,"wpr_ob":-22},
    "Industrials": {"rsi_os":30,"rsi_ob":68,"cci_os":-100,"cci_ob":108,"wpr_os":-80,"wpr_ob":-20},
}
DEFAULT_THRESH={"rsi_os":30,"rsi_ob":70,"cci_os":-100,"cci_ob":100,"wpr_os":-80,"wpr_ob":-20}
def get_thresh(s): return SECTOR_THRESHOLDS.get(s,DEFAULT_THRESH)

# ─────────────────────────────────────────────────────────
#  TECHNICAL INDICATOR MATH
# ─────────────────────────────────────────────────────────
def sma(s,n): return s.rolling(n,min_periods=n).mean()
def ema(s,n): return s.ewm(span=n,adjust=False,min_periods=n).mean()
def wma(s,n):
    w=np.arange(1,n+1)
    return s.rolling(n).apply(lambda x:np.dot(x,w)/w.sum(),raw=True)
def hma(s,n): return wma(2*wma(s,n//2)-wma(s,n),int(math.sqrt(n)))
def vwma(c,v,n): return (c*v).rolling(n).sum()/v.rolling(n).sum()
def ichimoku_base(h,l,n=26): return (h.rolling(n).max()+l.rolling(n).min())/2
def rsi_calc(c,n=14):
    d=c.diff();g=d.clip(lower=0);lo=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    al=lo.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))
def stoch_k(h,l,c,n=14,sm=3):
    ll=l.rolling(n).min();hh=h.rolling(n).max()
    return (100*(c-ll)/(hh-ll).replace(0,np.nan)).rolling(sm).mean()
def cci(h,l,c,n=20):
    tp=(h+l+c)/3;s=tp.rolling(n).mean()
    md=tp.rolling(n).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True)
    return (tp-s)/(0.015*md.replace(0,np.nan))
def adx_calc(h,l,c,n=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    pdm=h.diff().clip(lower=0);mdm=(-l.diff()).clip(lower=0)
    pdm=pdm.where(pdm>(-l.diff()).clip(lower=0),0.0)
    mdm=mdm.where(mdm>h.diff().clip(lower=0),0.0)
    atr=tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    pdi=100*pdm.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/atr.replace(0,np.nan)
    mdi=100*mdm.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,min_periods=n,adjust=False).mean(),pdi,mdi
def awesome_osc(h,l): mid=(h+l)/2; return sma(mid,5)-sma(mid,34)
def momentum_ind(c,n=10): return c-c.shift(n)
def macd_calc(c,f=12,sl=26,sg=9): ml=ema(c,f)-ema(c,sl); return ml,ema(ml,sg)
def stoch_rsi(c,rn=14,sn=14,k=3):
    rv=rsi_calc(c,rn);mn=rv.rolling(sn).min();mx=rv.rolling(sn).max()
    return ((rv-mn)/(mx-mn).replace(0,np.nan)).rolling(k).mean()
def williams_r(h,l,c,n=14):
    hh=h.rolling(n).max();ll=l.rolling(n).min()
    return -100*(hh-c)/(hh-ll).replace(0,np.nan)
def bbpower(h,l,c,n=13): e=ema(c,n); return (h-e)+(l-e)
def ult_osc(h,l,c,s=7,m=14,lg=28):
    bp=c-pd.concat([l,c.shift()],axis=1).min(axis=1)
    tr=pd.concat([h,c.shift()],axis=1).max(axis=1)-pd.concat([l,c.shift()],axis=1).min(axis=1)
    a=lambda p:bp.rolling(p).sum()/tr.rolling(p).sum().replace(0,np.nan)
    return 100*(4*a(s)+2*a(m)+a(lg))/7

def compute_score(hist,sector=""):
    if hist is None or len(hist)<100: return None,{}
    c,h,l,v=hist["Close"],hist["High"],hist["Low"],hist["Volume"]
    price=c.iloc[-1]; th=get_thresh(sector)
    signals,raw,mx={},0,0
    def sig_ma(val): return 0 if pd.isna(val) or val==0 else(1 if price>val else -1)
    for name,series in [("EMA 10",ema(c,10)),("SMA 10",sma(c,10)),("EMA 20",ema(c,20)),("SMA 20",sma(c,20)),
        ("EMA 30",ema(c,30)),("SMA 30",sma(c,30)),("EMA 50",ema(c,50)),("SMA 50",sma(c,50)),
        ("EMA 100",ema(c,100)),("SMA 100",sma(c,100)),("EMA 200",ema(c,200)),("SMA 200",sma(c,200)),
        ("Ichimoku",ichimoku_base(h,l,26)),("VWMA 20",vwma(c,v,20)),("HMA 9",hma(c,9))]:
        s=sig_ma(series.iloc[-1]); signals[name]=s; raw+=s; mx+=1
    def s_rsi(x): return 1 if x<th["rsi_os"] else(-1 if x>th["rsi_ob"] else 0)
    def s_cci(x): return 1 if x<th["cci_os"] else(-1 if x>th["cci_ob"] else 0)
    def s_wpr(x): return 1 if x<th["wpr_os"] else(-1 if x>th["wpr_ob"] else 0)
    def s_sign(x): return 0 if pd.isna(x) else(1 if x>0 else(-1 if x<0 else 0))
    adxv,pdi,mdi=adx_calc(h,l,c,14); ml,sig_l=macd_calc(c); w=1.5
    for name,s in [
        ("RSI 14",    s_rsi(rsi_calc(c,14).iloc[-1])),
        ("Stoch %K",  (1 if stoch_k(h,l,c).iloc[-1]<20 else(-1 if stoch_k(h,l,c).iloc[-1]>80 else 0))),
        ("CCI 20",    s_cci(cci(h,l,c).iloc[-1])),
        ("ADX 14",    (1 if pdi.iloc[-1]>mdi.iloc[-1] else -1) if adxv.iloc[-1]>25 else 0),
        ("Awe. Osc.", s_sign(awesome_osc(h,l).iloc[-1])),
        ("Momentum",  s_sign(momentum_ind(c).iloc[-1])),
        ("MACD",      s_sign(ml.iloc[-1]-sig_l.iloc[-1])),
        ("Stoch RSI", (1 if stoch_rsi(c).iloc[-1]<0.2 else(-1 if stoch_rsi(c).iloc[-1]>0.8 else 0))),
        ("Williams %R",s_wpr(williams_r(h,l,c).iloc[-1])),
        ("Bull/Bear", s_sign(bbpower(h,l,c).iloc[-1])),
        ("Ult. Osc.", (1 if ult_osc(h,l,c).iloc[-1]<30 else(-1 if ult_osc(h,l,c).iloc[-1]>70 else 0))),
    ]:
        signals[name]=s; raw+=s*w; mx+=w
    return round(max(0,min(100,((raw+mx)/(2*mx))*100)),1), signals

# ─────────────────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300,show_spinner=False)
def get_info(t):
    """
    Robust info fetch — combines yf.fast_info (reliable price/cap data)
    with yf.info (fundamentals, analyst data, descriptions).
    Newer yfinance versions moved currentPrice/marketCap out of .info,
    so we normalise everything into one consistent dict.
    """
    result = {}
    ticker_obj = yf.Ticker(t)

    # ── fast_info: reliable price + market data ──────────
    try:
        fi = ticker_obj.fast_info
        # fast_info uses attribute access, not dict keys
        result["currentPrice"]    = getattr(fi, "last_price",      None)
        result["marketCap"]       = getattr(fi, "market_cap",       None)
        result["sharesOutstanding"]= getattr(fi, "shares",          None)
        result["fiftyTwoWeekHigh"] = getattr(fi, "year_high",       None)
        result["fiftyTwoWeekLow"]  = getattr(fi, "year_low",        None)
        result["currency"]         = getattr(fi, "currency",        None)
    except Exception:
        pass

    # ── .info: fundamentals, analyst data, descriptions ──
    try:
        info = ticker_obj.info or {}
        # Only update keys not already set by fast_info (fast_info is more reliable)
        for k, v in info.items():
            if k not in result or result[k] is None:
                result[k] = v
        # Ensure currentPrice fallback chain
        if not result.get("currentPrice"):
            result["currentPrice"] = (info.get("currentPrice")
                                   or info.get("regularMarketPrice")
                                   or info.get("previousClose"))
        # Ensure marketCap fallback
        if not result.get("marketCap"):
            result["marketCap"] = info.get("marketCap")
    except Exception:
        pass

    return result
@st.cache_data(ttl=300,show_spinner=False)
def get_hist(t,period="2y",interval="1d"):
    try:
        h=yf.Ticker(t).history(period=period,interval=interval)
        return h if len(h)>10 else None
    except: return None

def fmt_cap(mc):
    if not mc: return "—"
    if mc>=1e12: return f"${mc/1e12:.1f}T"
    if mc>=1e9:  return f"${mc/1e9:.1f}B"
    return f"${mc/1e6:.0f}M"
def score_color(s):
    if s is None: return "#6b6e6c"
    if s>=70: return "#4caf7d"
    if s>=55: return "#8bc98b"
    if s>=45: return "#5bc8c8"
    if s>=30: return "#e07b5c"
    return "#e05c5c"
def score_label(s):
    if s is None: return "N/A"
    if s>=70: return "Strong Buy"
    if s>=55: return "Buy"
    if s>=45: return "Neutral"
    if s>=30: return "Sell"
    return "Strong Sell"
def analyst_label(rec):
    return {"strong_buy":"Strong Buy","buy":"Buy","hold":"Hold",
            "underperform":"Underperform","sell":"Sell"}.get((rec or "").lower().replace(" ","_"),rec or "—")

# ─────────────────────────────────────────────────────────
#  PLOTLY THEME DEFAULTS
# ─────────────────────────────────────────────────────────
CHART_LAYOUT = dict(paper_bgcolor="#111210",plot_bgcolor="#111210",
    font=dict(color="#6b6e6c",family="DM Mono"),
    xaxis=dict(showgrid=False,color="#6b6e6c",tickfont=dict(size=10)),
    yaxis=dict(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",tickfont=dict(size=10)),
    legend=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(size=10,color="#a0a8a4")),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(color="#e8e4dc",family="DM Mono",size=11)))

# ─────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────
def section_header(text,sub=""):
    st.markdown(f"""<div style="font-size:9px;letter-spacing:.2em;text-transform:uppercase;
        color:#6b6e6c;margin-bottom:{'4' if sub else '12'}px;padding-top:8px;">{text}</div>
        {'<div style="font-size:11px;color:#6b6e6c;margin-bottom:12px;">'+sub+'</div>' if sub else ''}""",
        unsafe_allow_html=True)

def metric_card(label,value,sub="",color="#c8953a"):
    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid {color};
        border-radius:3px;padding:14px 18px;margin-bottom:4px;">
        <div style="font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:#6b6e6c;margin-bottom:5px;">{label}</div>
        <div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:300;color:#e8e4dc;">{value}</div>
        <div style="font-size:10px;color:#6b6e6c;margin-top:3px;">{sub}</div>
    </div>""",unsafe_allow_html=True)

def info_card(title,body,color="#c8953a"):
    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid {color};
        padding:16px 20px;border-radius:3px;margin-bottom:12px;">
        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:{color};margin-bottom:8px;">{title}</div>
        <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">{body}</div>
    </div>""",unsafe_allow_html=True)

def score_gauge(score,ticker):
    color=score_color(score); label=score_label(score)
    fig=go.Figure(go.Indicator(mode="gauge+number",value=score or 0,
        number={"suffix":"/100","font":{"size":26,"color":"#e8e4dc","family":"Fraunces, serif"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#2a2c2b","tickfont":{"color":"#6b6e6c","size":10}},
               "bar":{"color":color,"thickness":0.3},"bgcolor":"#181917","bordercolor":"#2a2c2b",
               "steps":[{"range":[0,30],"color":"#1a1210"},{"range":[30,45],"color":"#1a1512"},
                        {"range":[45,55],"color":"#141a1a"},{"range":[55,70],"color":"#121a12"},{"range":[70,100],"color":"#101810"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":score or 0}},
        title={"text":f"<b>{ticker}</b><br><span style='font-size:12px;color:{color}'>{label}</span>",
               "font":{"color":"#e8e4dc","family":"Fraunces, serif","size":15}}))
    fig.update_layout(height=220,margin=dict(l=20,r=20,t=36,b=10),paper_bgcolor="#111210",font_color="#e8e4dc")
    return fig

def price_chart(hist,ticker):
    if hist is None: return None
    c=hist["Close"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist.index,y=c,name="Price",line=dict(color="#c8953a",width=2)))
    fig.add_trace(go.Scatter(x=hist.index,y=ema(c,20),name="EMA 20",line=dict(color="#5bc8c8",width=1,dash="dot"),opacity=0.7))
    fig.add_trace(go.Scatter(x=hist.index,y=ema(c,50),name="EMA 50",line=dict(color="#8b7ed8",width=1,dash="dot"),opacity=0.7))
    fig.add_trace(go.Scatter(x=hist.index,y=ema(c,200),name="EMA 200",line=dict(color="#e05c5c",width=1,dash="dot"),opacity=0.7))
    layout=CHART_LAYOUT.copy()
    layout.update(height=280,margin=dict(l=0,r=0,t=24,b=0),
        yaxis=dict(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",tickformat="$.2f",tickfont=dict(size=10)),
        title=dict(text=f"{ticker} — 1 Year",font=dict(size=13,color="#e8e4dc",family="Fraunces, serif"),x=0))
    fig.update_layout(**layout)
    return fig

def signals_chart(signals):
    vals=[signals.get(n,0) for n in ALL_INDICATORS]
    colors=["#4caf7d" if v==1 else("#e05c5c" if v==-1 else "#2a2c2b") for v in vals]
    labels=["BUY" if v==1 else("SELL" if v==-1 else "NEU") for v in vals]
    fig=go.Figure(go.Bar(x=ALL_INDICATORS,y=[1]*26,marker_color=colors,text=labels,textposition="inside",
        textfont=dict(size=9,color="#0b0a09",family="DM Mono"),hovertemplate="%{x}: %{text}<extra></extra>"))
    fig.add_vline(x=14.5,line_color="#2a2c2b",line_width=1,line_dash="dot")
    fig.add_annotation(x=7,y=1.05,text="MOVING AVERAGES",showarrow=False,font=dict(size=9,color="#6b6e6c",family="DM Mono"),yref="paper")
    fig.add_annotation(x=20,y=1.05,text="OSCILLATORS",showarrow=False,font=dict(size=9,color="#6b6e6c",family="DM Mono"),yref="paper")
    fig.update_layout(height=180,margin=dict(l=0,r=0,t=30,b=40),paper_bgcolor="#111210",plot_bgcolor="#111210",
        showlegend=False,xaxis=dict(showgrid=False,tickfont=dict(size=8,color="#6b6e6c",family="DM Mono"),tickangle=-45),yaxis=dict(visible=False))
    return fig

def style_df(df):
    def c_score(v):
        try: return f"color:{score_color(float(str(v)))};font-weight:500"
        except: return ""
    def c_up(v):
        if v=="—": return ""
        try: n=float(str(v).replace("%","").replace("+","")); return "color:#4caf7d" if n>=0 else "color:#e05c5c"
        except: return ""
    def c_con(v):
        return {"Strong Buy":"color:#4caf7d","Buy":"color:#8bc98b","Hold":"color:#5bc8c8",
                "Underperform":"color:#e07b5c","Sell":"color:#e05c5c"}.get(v,"")
    styled=df.style
    for col,fn in [("Score",c_score),("Upside",c_up),("Consensus",c_con),("Signal",c_con)]:
        if col in df.columns: styled=styled.applymap(fn,subset=[col])
    return styled.set_properties(**{"background-color":"#111210","color":"#e8e4dc","border-color":"#1f2020",
        "font-family":"DM Mono, monospace","font-size":"12px"}).set_table_styles([
        {"selector":"th","props":[("background-color","#0b0a09"),("color","#6b6e6c"),("font-size","9px"),
         ("letter-spacing","0.15em"),("text-transform","uppercase"),("font-family","DM Mono, monospace"),
         ("border-color","#1f2020"),("padding","10px 12px")]},
        {"selector":"td","props":[("padding","10px 12px"),("border-color","#1f2020")]},
        {"selector":"tr:hover td","props":[("background-color","#181917")]},])

def model_ticker_input(key):
    c1,c2=st.columns([3,1])
    with c1: t=st.text_input("",placeholder="Enter ticker e.g. AAPL",label_visibility="collapsed",key=f"mt_{key}").strip().upper()
    with c2: run=st.button("Run Model →",key=f"mb_{key}",use_container_width=True)
    return t, run

# ─────────────────────────────────────────────────────────
#  FINANCIAL MODELS
# ─────────────────────────────────────────────────────────

# ── 1. STAGE ANALYSIS ────────────────────────────────────
def model_stage_analysis():
    info_card("Stan Weinstein's Stage Analysis",
        "Identifies where a stock sits in its long-term cycle using weekly price data and the 40-week moving average. "
        "Stage 1 = Basing, Stage 2 = Advancing (best time to buy), Stage 3 = Topping, Stage 4 = Declining.")
    ticker, run = model_ticker_input("stage")
    if not run or not ticker: return

    with st.spinner(f"Loading weekly data for {ticker}…"):
        hist = get_hist(ticker, period="3y", interval="1wk")
    if hist is None or len(hist) < 45:
        st.error("Not enough weekly data. Try a larger-cap stock."); return

    c = pd.Series(hist["Close"].values.flatten(), index=hist.index)
    ma40 = sma(c, 40)
    ma10 = sma(c, 10)

    # Stage detection on last 8 weeks
    recent_c   = c.iloc[-8:]
    recent_ma  = ma40.iloc[-8:]
    ma_slope   = (ma40.iloc[-1] - ma40.iloc[-8]) / ma40.iloc[-8] * 100
    price_vs_ma= (c.iloc[-1] - ma40.iloc[-1]) / ma40.iloc[-1] * 100

    if c.iloc[-1] > ma40.iloc[-1] and ma_slope > 0.3:
        stage, stage_color, stage_desc = 2, "#4caf7d", "Advancing — price above rising 40-week MA. Prime buy zone."
    elif c.iloc[-1] > ma40.iloc[-1] and ma_slope <= 0.3:
        stage, stage_color, stage_desc = 3, "#e07b5c", "Topping — price near MA from above, MA flattening. Caution."
    elif c.iloc[-1] < ma40.iloc[-1] and ma_slope < -0.3:
        stage, stage_color, stage_desc = 4, "#e05c5c", "Declining — price below falling 40-week MA. Avoid."
    else:
        stage, stage_color, stage_desc = 1, "#5bc8c8", "Basing — price near flat MA. Accumulation zone forming."

    # Metrics row
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Current Stage", f"Stage {stage}", stage_desc[:40]+"…", stage_color)
    with c2: metric_card("Price vs 40W MA", f"{price_vs_ma:+.1f}%", "above = bullish", "#c8953a")
    with c3: metric_card("40W MA Slope", f"{ma_slope:+.2f}%", "8-week change", "#5bc8c8")
    with c4: metric_card("40W MA", f"${ma40.iloc[-1]:.2f}", "key support/resistance", "#6b6e6c")

    st.markdown("<br>",unsafe_allow_html=True)

    # Chart
    fig = go.Figure()
    # Stage background shading
    stage_bands = {"Stage 1":(1,"rgba(91,200,200,0.13)"),"Stage 2":(2,"rgba(76,175,125,0.13)"),
                   "Stage 3":(3,"rgba(224,123,92,0.13)"),"Stage 4":(4,"rgba(224,92,92,0.13)")}
    fig.add_trace(go.Scatter(x=hist.index,y=c,name="Price (Weekly)",line=dict(color="#c8953a",width=2)))
    fig.add_trace(go.Scatter(x=hist.index,y=ma40,name="40-Week MA",line=dict(color="#e8c97a",width=2,dash="dash")))
    fig.add_trace(go.Scatter(x=hist.index,y=ma10,name="10-Week MA",line=dict(color="#5bc8c8",width=1,dash="dot"),opacity=0.7))
    # Mark current stage — rgba() required, Plotly rejects 8-digit hex
    def _hex_to_rgba(hx, alpha=0.12):
        r,g,b = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
        return f"rgba({r},{g},{b},{alpha})"
    fig.add_hrect(y0=ma40.iloc[-1]*0.97, y1=ma40.iloc[-1]*1.03,
        fillcolor=_hex_to_rgba(stage_color), line_width=0,
        annotation=dict(text=f"Stage {stage} Zone", font=dict(color=stage_color, size=10)))
    layout=CHART_LAYOUT.copy()
    layout.update(height=340,margin=dict(l=0,r=0,t=24,b=0),
        yaxis=dict(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",tickformat="$.2f",tickfont=dict(size=10)),
        title=dict(text=f"{ticker} — Weinstein Stage Analysis (Weekly)",
                   font=dict(size=13,color="#e8e4dc",family="Fraunces, serif"),x=0))
    fig.update_layout(**layout)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid {stage_color};
        padding:16px 20px;border-radius:3px;margin-top:8px;">
        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:{stage_color};margin-bottom:8px;">Stage {stage} Interpretation</div>
        <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">{stage_desc}<br><br>
        <b style="color:#e8e4dc;">Weinstein's Rule:</b> Only buy in Stage 2. Never average down in Stage 4.
        Stage 1 requires patience — wait for the breakout above the 40-week MA on volume.</div>
    </div>""",unsafe_allow_html=True)

# ── 2. MANSFIELD RELATIVE STRENGTH ───────────────────────
def model_mansfield_rs():
    info_card("Mansfield Relative Strength",
        "Measures whether a stock is outperforming or underperforming the S&P 500. "
        "A positive MRS reading (above zero) is required for a valid Stage 2 breakout per Weinstein. "
        "Formula: (Stock/SPY ratio) vs its own 52-week average — normalised to zero.")
    ticker, run = model_ticker_input("mans")
    if not run or not ticker: return

    with st.spinner(f"Comparing {ticker} vs SPY…"):
        h_stock = get_hist(ticker, period="2y", interval="1wk")
        h_spy   = get_hist("SPY",  period="2y", interval="1wk")

    if h_stock is None or h_spy is None:
        st.error("Could not load data."); return

    c_s = pd.Series(h_stock["Close"].values.flatten(), index=h_stock.index)
    c_m = pd.Series(h_spy["Close"].values.flatten(), index=h_spy.index)
    common = c_s.index.intersection(c_m.index)
    c_s, c_m = c_s.loc[common], c_m.loc[common]

    ratio   = c_s / c_m
    ma52    = sma(ratio, 52)
    mrs     = ((ratio / ma52) - 1) * 100
    current = mrs.iloc[-1]

    c1,c2,c3 = st.columns(3)
    color = "#4caf7d" if current>0 else "#e05c5c"
    with c1: metric_card("Mansfield RS", f"{current:+.1f}", "positive = outperforming SPY", color)
    with c2: metric_card("Stock vs SPY (1Y)", f"{((c_s.iloc[-1]/c_s.iloc[-53]-1)*100):+.1f}%", f"{ticker} 1-year return", "#c8953a")
    with c3: metric_card("SPY (1Y)", f"{((c_m.iloc[-1]/c_m.iloc[-53]-1)*100):+.1f}%","S&P 500 benchmark","#6b6e6c")

    st.markdown("<br>",unsafe_allow_html=True)

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.04)

    # Normalised price comparison
    norm_s = (c_s/c_s.iloc[0])*100
    norm_m = (c_m/c_m.iloc[0])*100
    fig.add_trace(go.Scatter(x=common,y=norm_s,name=ticker,line=dict(color="#c8953a",width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=common,y=norm_m,name="SPY",line=dict(color="#5bc8c8",width=1,dash="dot")),row=1,col=1)

    # MRS line
    mrs_colors=["#4caf7d" if v>=0 else "#e05c5c" for v in mrs.fillna(0)]
    fig.add_trace(go.Bar(x=common,y=mrs,name="Mansfield RS",marker_color=mrs_colors,opacity=0.7),row=2,col=1)
    fig.add_hline(y=0,line_color="#6b6e6c",line_width=1,row=2,col=1)

    fig.update_layout(height=400,margin=dict(l=0,r=0,t=24,b=0),paper_bgcolor="#111210",plot_bgcolor="#111210",
        font=dict(color="#6b6e6c",family="DM Mono"),
        legend=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(size=10,color="#a0a8a4")),
        title=dict(text=f"{ticker} vs SPY — Mansfield Relative Strength",
                   font=dict(size=13,color="#e8e4dc",family="Fraunces, serif"),x=0),
        hovermode="x unified",hoverlabel=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(color="#e8e4dc",family="DM Mono",size=11)))
    fig.update_xaxes(showgrid=False,color="#6b6e6c",tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",tickfont=dict(size=10))
    fig.update_yaxes(ticksuffix="%",row=1,col=1)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    interp = (f"{ticker} is <b style='color:#4caf7d'>outperforming</b> the S&P 500 (MRS: {current:+.1f}). "
              "This is a prerequisite for a valid Stage 2 breakout per Weinstein's methodology." if current>0
              else f"{ticker} is <b style='color:#e05c5c'>underperforming</b> the S&P 500 (MRS: {current:+.1f}). "
              "Even if price looks strong, relative weakness vs the market is a red flag for long entries.")
    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid {color};
        padding:16px 20px;border-radius:3px;margin-top:8px;">
        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:{color};margin-bottom:8px;">Relative Strength Reading</div>
        <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">{interp}</div>
    </div>""",unsafe_allow_html=True)

# ── 3. MEAN REVERSION (BB + RSI) ─────────────────────────
def model_mean_reversion():
    info_card("Mean Reversion — Bollinger Bands + RSI",
        "Identifies 'rubber band' trades where price has stretched too far from its mean. "
        "Bollinger Bands show price deviation from the 20-day SMA. RSI below 30 or above 70 confirms the extreme. "
        "When both agree, a reversion is statistically likely.")
    ticker, run = model_ticker_input("bbr")
    if not run or not ticker: return

    with st.spinner(f"Running mean reversion analysis for {ticker}…"):
        hist = get_hist(ticker)
    if hist is None: st.error("Could not load data."); return

    c = pd.Series(hist["Close"].values.flatten(), index=hist.index)

    # Bollinger Bands
    bb_mid = sma(c, 20)
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid * 100  # bandwidth %

    rsi_series = rsi_calc(c, 14)
    current_rsi = rsi_series.iloc[-1]
    price = c.iloc[-1]
    pct_b = (price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])  # 0=lower band, 1=upper band

    # Signal
    if pct_b < 0.1 and current_rsi < 35:
        sig, sig_color, sig_text = "Oversold — Reversion Buy", "#4caf7d", "Price at/below lower band with RSI oversold. Classic mean reversion long setup."
    elif pct_b > 0.9 and current_rsi > 65:
        sig, sig_color, sig_text = "Overbought — Reversion Sell", "#e05c5c", "Price at/above upper band with RSI overbought. Rubber band stretched — potential pullback."
    elif bb_width.iloc[-1] < bb_width.rolling(50).mean().iloc[-1] * 0.8:
        sig, sig_color, sig_text = "Bollinger Squeeze", "#c8953a", "Bands are unusually narrow — a big move is building. Direction unknown, watch for breakout."
    else:
        sig, sig_color, sig_text = "Neutral", "#5bc8c8", "Price within normal range. No extreme mean reversion signal present."

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Signal", sig, "", sig_color)
    with c2: metric_card("RSI (14)", f"{current_rsi:.1f}", "30=OS / 70=OB", "#c8953a")
    with c3: metric_card("%B", f"{pct_b:.2f}", "0=lower band, 1=upper", "#5bc8c8")
    with c4: metric_card("BB Width", f"{bb_width.iloc[-1]:.1f}%", "narrower = squeeze", "#6b6e6c")

    st.markdown("<br>",unsafe_allow_html=True)

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=hist.index,y=bb_upper,name="Upper Band",line=dict(color="#e05c5c",width=1,dash="dot"),opacity=0.6),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=bb_lower,name="Lower Band",fill="tonexty",fillcolor="rgba(91,200,200,0.03)",line=dict(color="#4caf7d",width=1,dash="dot"),opacity=0.6),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=bb_mid,name="20-Day SMA",line=dict(color="#6b6e6c",width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=c,name="Price",line=dict(color="#c8953a",width=2)),row=1,col=1)
    rsi_colors=["#4caf7d" if v<30 else("#e05c5c" if v>70 else "#5bc8c8") for v in rsi_series.fillna(50)]
    fig.add_trace(go.Scatter(x=hist.index,y=rsi_series,name="RSI 14",line=dict(color="#c8953a",width=1.5)),row=2,col=1)
    fig.add_hline(y=70,line_color="#e05c5c",line_width=1,line_dash="dot",row=2,col=1)
    fig.add_hline(y=30,line_color="#4caf7d",line_width=1,line_dash="dot",row=2,col=1)
    fig.add_hrect(y0=30,y1=70,fillcolor="rgba(91,200,200,0.03)",line_width=0,row=2,col=1)
    fig.update_layout(height=420,margin=dict(l=0,r=0,t=24,b=0),paper_bgcolor="#111210",plot_bgcolor="#111210",
        font=dict(color="#6b6e6c",family="DM Mono"),
        legend=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(size=10,color="#a0a8a4")),
        title=dict(text=f"{ticker} — Bollinger Bands + RSI Mean Reversion",
                   font=dict(size=13,color="#e8e4dc",family="Fraunces, serif"),x=0),
        hovermode="x unified",hoverlabel=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(color="#e8e4dc",family="DM Mono",size=11)))
    fig.update_xaxes(showgrid=False,color="#6b6e6c",tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",tickfont=dict(size=10))
    fig.update_yaxes(tickformat="$.2f",row=1,col=1)
    fig.update_yaxes(range=[0,100],row=2,col=1)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid {sig_color};
        padding:16px 20px;border-radius:3px;">
        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:{sig_color};margin-bottom:6px;">Reading</div>
        <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">{sig_text}</div>
    </div>""",unsafe_allow_html=True)

# ── 4. ELLIOTT WAVE + FIBONACCI ──────────────────────────
def model_elliott_wave():
    info_card("Elliott Wave & Fibonacci Retracements",
        "Maps significant price swings using peak/trough detection, then draws Fibonacci retracement levels "
        "between the most recent major swing high and low. Elliott Wave labels are approximate — they identify "
        "the likely wave structure based on swing count and momentum.")
    ticker, run = model_ticker_input("elliott")
    if not run or not ticker: return
    if not HAS_SCIPY:
        st.warning("scipy not installed. Run: pip install scipy"); return

    with st.spinner(f"Detecting swing points for {ticker}…"):
        hist = get_hist(ticker, period="1y")
    if hist is None: st.error("Could not load data."); return

    c = pd.Series(hist["Close"].values.flatten(), index=hist.index).values
    dates = hist.index

    # Find peaks and troughs with adaptive order
    order = max(5, len(c)//20)
    peak_idx = argrelextrema(c, np.greater, order=order)[0]
    trough_idx = argrelextrema(c, np.less, order=order)[0]

    # Combine and sort all swing points
    swings = sorted(
        [(i,"H",c[i]) for i in peak_idx] + [(i,"L",c[i]) for i in trough_idx],
        key=lambda x: x[0]
    )

    # Fibonacci levels from the most significant recent swing
    if len(swings) >= 2:
        # Find the largest swing range in recent history (last 40% of data)
        cutoff = int(len(c)*0.6)
        recent_peaks = [s for s in swings if s[0]>=cutoff and s[1]=="H"]
        recent_troughs = [s for s in swings if s[0]>=cutoff and s[1]=="L"]
        if recent_peaks and recent_troughs:
            swing_high = max(recent_peaks, key=lambda x:x[2])
            swing_low  = min(recent_troughs, key=lambda x:x[2])
            fib_high, fib_low = swing_high[2], swing_low[2]
            fib_range = fib_high - fib_low
            fib_levels = {
                "0%":   fib_low,
                "23.6%":fib_low + 0.236*fib_range,
                "38.2%":fib_low + 0.382*fib_range,
                "50%":  fib_low + 0.500*fib_range,
                "61.8%":fib_low + 0.618*fib_range,
                "78.6%":fib_low + 0.786*fib_range,
                "100%": fib_high,
            }
        else:
            fib_levels = {}
            fib_high = fib_low = None
    else:
        fib_levels = {}
        fib_high = fib_low = None

    # Current price location relative to fib levels
    current_price = c[-1]
    if fib_levels:
        nearest = min(fib_levels.items(), key=lambda x: abs(x[1]-current_price))
        fib_note = f"Price near {nearest[0]} Fibonacci level (${nearest[1]:.2f})"
    else:
        fib_note = "Insufficient swing data"

    # Wave count estimate
    wave_count = len(swings)
    wave_phase = "impulse (5-wave)" if wave_count % 5 <= 2 else "corrective (ABC)"

    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Swing Points", str(len(swings)), "peaks + troughs detected","#c8953a")
    with c2: metric_card("Likely Phase", wave_phase, "estimated from swing count","#5bc8c8")
    with c3: metric_card("Nearest Fib", nearest[0] if fib_levels else "—", fib_note[:35],"#c8953a")

    st.markdown("<br>",unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=c,name="Price",line=dict(color="#c8953a",width=2)))

    # Mark swing highs and lows
    for idx,stype,price_val in swings[-20:]:  # last 20 swings
        color_s = "#4caf7d" if stype=="H" else "#e05c5c"
        label_s = "H" if stype=="H" else "L"
        fig.add_trace(go.Scatter(x=[dates[idx]],y=[price_val],mode="markers+text",
            marker=dict(color=color_s,size=8,symbol="triangle-up" if stype=="H" else "triangle-down"),
            text=[label_s],textposition="top center" if stype=="H" else "bottom center",
            textfont=dict(size=9,color=color_s),showlegend=False,
            hovertemplate=f"{label_s}: ${{price_val:.2f}}<extra></extra>"))

    # Fibonacci lines
    fib_colors={"0%":"#6b6e6c","23.6%":"#5bc8c8","38.2%":"#c8953a",
                "50%":"#e8c97a","61.8%":"#c8953a","78.6%":"#5bc8c8","100%":"#6b6e6c"}
    for label,level in fib_levels.items():
        fig.add_hline(y=level,line_color=fib_colors.get(label,"#6b6e6c"),line_width=1,line_dash="dot")
        fig.add_annotation(x=1.01,y=level,xref="paper",yref="y",
            text=f"{label} ${level:.2f}",showarrow=False,xanchor="left",
            font=dict(color=fib_colors.get(label,"#6b6e6c"),size=9))

    layout=CHART_LAYOUT.copy()
    layout.update(height=400,margin=dict(l=0,r=60,t=24,b=0),
        yaxis=dict(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",tickformat="$.2f",tickfont=dict(size=10)),
        title=dict(text=f"{ticker} — Elliott Wave Swings + Fibonacci Retracements",
                   font=dict(size=13,color="#e8e4dc",family="Fraunces, serif"),x=0),
        showlegend=False)
    fig.update_layout(**layout)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid #c8953a;
        padding:16px 20px;border-radius:3px;">
        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#c8953a;margin-bottom:6px;">How to Read This</div>
        <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">
            Green triangles (▲) = swing highs · Red triangles (▼) = swing lows.<br>
            Fibonacci levels are drawn from the most significant recent swing. The 38.2%, 50%, and 61.8% levels are the strongest support/resistance zones.<br>
            <b style="color:#e8e4dc;">Current position:</b> {fib_note}. Estimated wave phase: <b style="color:#e8e4dc;">{wave_phase}</b>.
        </div>
    </div>""",unsafe_allow_html=True)

# ── 5. VOLUME PROFILE + ANCHORED VWAP ────────────────────
def model_volume_profile():
    info_card("Volume Profile + Anchored VWAP",
        "Volume Profile shows the price levels where the most shares have traded — these become support and resistance. "
        "Point of Control (POC) = the single price level with the most volume. "
        "Anchored VWAP = volume-weighted average price from a chosen start date.")

    ticker, run = model_ticker_input("vp")
    if not run or not ticker: return

    with st.spinner(f"Building volume profile for {ticker}…"):
        hist = get_hist(ticker, period="1y")
    if hist is None: st.error("Could not load data."); return

    c  = pd.Series(hist["Close"].values.flatten(), index=hist.index)
    h  = pd.Series(hist["High"].values.flatten(), index=hist.index)
    l  = pd.Series(hist["Low"].values.flatten(), index=hist.index)
    v  = pd.Series(hist["Volume"].values.flatten(), index=hist.index)

    # ── Volume Profile ──
    n_bins  = 40
    p_min, p_max = l.min(), h.max()
    bins    = np.linspace(p_min, p_max, n_bins+1)
    bin_mid = (bins[:-1]+bins[1:])/2
    vol_at_price = np.zeros(n_bins)

    for i in range(len(hist)):
        lo, hi, vol = l.iloc[i], h.iloc[i], v.iloc[i]
        for b in range(n_bins):
            overlap = max(0, min(hi, bins[b+1]) - max(lo, bins[b]))
            rng     = hi - lo if hi != lo else 1
            vol_at_price[b] += vol * overlap / rng

    poc_idx  = np.argmax(vol_at_price)
    poc_price= bin_mid[poc_idx]

    # Value Area (70% of total volume around POC)
    total_vol = vol_at_price.sum()
    va_vol    = 0
    va_indices= [poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while va_vol < total_vol * 0.70 and (lo_i > 0 or hi_i < n_bins-1):
        add_hi = vol_at_price[hi_i+1] if hi_i < n_bins-1 else 0
        add_lo = vol_at_price[lo_i-1] if lo_i > 0 else 0
        if add_hi >= add_lo and hi_i < n_bins-1:
            hi_i += 1; va_vol += vol_at_price[hi_i]; va_indices.append(hi_i)
        elif lo_i > 0:
            lo_i -= 1; va_vol += vol_at_price[lo_i]; va_indices.append(lo_i)
        else:
            break
    val_price = bin_mid[lo_i]   # Value Area Low
    vah_price = bin_mid[hi_i]   # Value Area High

    # ── Anchored VWAP ──
    typical = (h + l + c) / 3
    cum_tv   = (typical * v).cumsum()
    cum_v    = v.cumsum()
    vwap     = cum_tv / cum_v

    current_price = c.iloc[-1]
    poc_dist = (current_price - poc_price) / poc_price * 100

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Point of Control", f"${poc_price:.2f}", "highest volume price level","#c8953a")
    with c2: metric_card("Value Area High", f"${vah_price:.2f}", "70% value area ceiling","#4caf7d")
    with c3: metric_card("Value Area Low", f"${val_price:.2f}", "70% value area floor","#e05c5c")
    with c4: metric_card("Price vs POC", f"{poc_dist:+.1f}%",
                         "above POC = bullish" if poc_dist>0 else "below POC = bearish",
                         "#4caf7d" if poc_dist>0 else "#e05c5c")

    st.markdown("<br>",unsafe_allow_html=True)

    # Chart: price + VWAP on left, volume profile histogram on right
    fig = make_subplots(rows=1,cols=2,column_widths=[0.78,0.22],shared_yaxes=True,
                        horizontal_spacing=0.01)

    fig.add_trace(go.Scatter(x=hist.index,y=c,name="Price",line=dict(color="#c8953a",width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=vwap,name="Anchored VWAP",line=dict(color="#e8c97a",width=1.5,dash="dash")),row=1,col=1)
    fig.add_hrect(y0=val_price,y1=vah_price,fillcolor="rgba(91,200,200,0.13)",line_width=0,row=1,col=1)
    fig.add_hline(y=poc_price,line_color="#c8953a",line_width=2,line_dash="solid",row=1,col=1)
    fig.add_annotation(x=0.77,y=poc_price,xref="paper",yref="y",
        text=f"POC ${poc_price:.2f}",showarrow=False,xanchor="left",
        font=dict(color="#c8953a",size=9))
    fig.add_hline(y=vah_price,line_color="#4caf7d",line_width=1,line_dash="dot",row=1,col=1)
    fig.add_hline(y=val_price,line_color="#e05c5c",line_width=1,line_dash="dot",row=1,col=1)

    # Volume histogram bars
    bar_colors=["#c8953a" if i==poc_idx else("rgba(91,200,200,0.5)" if lo_i<=i<=hi_i else "rgba(42,44,43,0.5)") for i in range(n_bins)]
    fig.add_trace(go.Bar(x=vol_at_price,y=bin_mid,orientation="h",name="Volume Profile",
                         marker_color=bar_colors,showlegend=False),row=1,col=2)

    fig.update_layout(height=400,margin=dict(l=0,r=0,t=24,b=0),paper_bgcolor="#111210",plot_bgcolor="#111210",
        font=dict(color="#6b6e6c",family="DM Mono"),
        legend=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(size=10,color="#a0a8a4")),
        title=dict(text=f"{ticker} — Volume Profile + Anchored VWAP (1 Year)",
                   font=dict(size=13,color="#e8e4dc",family="Fraunces, serif"),x=0),
        hovermode="x",hoverlabel=dict(bgcolor="#181917",bordercolor="#2a2c2b",font=dict(color="#e8e4dc",family="DM Mono",size=11)),
        bargap=0.05)
    fig.update_xaxes(showgrid=False,color="#6b6e6c",tickfont=dict(size=10),row=1,col=1)
    fig.update_xaxes(showgrid=False,visible=False,row=1,col=2)
    fig.update_yaxes(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",tickformat="$.2f",tickfont=dict(size=10))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    pos = ("above" if current_price>poc_price else "below")
    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid #c8953a;
        padding:16px 20px;border-radius:3px;">
        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#c8953a;margin-bottom:6px;">How to Use This</div>
        <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">
            The <b style="color:#c8953a;">orange horizontal bar</b> is the Point of Control — the price where the most shares traded. It acts as a magnet.
            The <b style="color:#5bc8c8;">blue shaded zone</b> is the Value Area (70% of volume). Price inside the zone = fair value. Price outside = extended.<br><br>
            <b style="color:#e8e4dc;">Current:</b> Price is {pos} the POC (${poc_price:.2f}) by {abs(poc_dist):.1f}%.
            Value Area: ${val_price:.2f} – ${vah_price:.2f}.
        </div>
    </div>""",unsafe_allow_html=True)

# ── 6. DCF ────────────────────────────────────────────────
def model_dcf():
    info_card("Discounted Cash Flow (DCF)",
        "Estimates a stock's intrinsic value by projecting future free cash flows and discounting them back "
        "to today's dollars. If intrinsic value > current price, the stock may be undervalued. "
        "Uses yfinance cash flow statements and balance sheet data.")

    ticker, run = model_ticker_input("dcf")

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("""<div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;margin-bottom:10px;">Assumptions</div>""",unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4)
    with a1: wacc     = st.number_input("WACC (%)",  min_value=1.0, max_value=30.0, value=10.0, step=0.5, key="dcf_wacc") / 100
    with a2: g_rate   = st.number_input("Growth Rate Yr 1–5 (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, key="dcf_g") / 100
    with a3: term_g   = st.number_input("Terminal Growth (%)", min_value=0.0, max_value=5.0, value=2.5, step=0.5, key="dcf_tg") / 100
    with a4: years    = st.number_input("Projection Years", min_value=3, max_value=10, value=5, step=1, key="dcf_yr")

    if not run or not ticker: return

    with st.spinner(f"Loading financials for {ticker}…"):
        t = yf.Ticker(ticker)
        info = get_info(ticker)
        try:
            cf   = t.cashflow
            bs   = t.balance_sheet
        except:
            st.error("Could not load financial statements."); return

    # Extract FCF
    fcf = None
    try:
        op_cf  = cf.loc["Operating Cash Flow"].iloc[0]
        capex  = cf.loc["Capital Expenditure"].iloc[0]
        fcf    = op_cf + capex  # capex is negative in yfinance
    except:
        pass
    if fcf is None or fcf <= 0:
        st.error(f"Could not extract positive Free Cash Flow for {ticker}. DCF requires profitable FCF."); return

    # Extract net debt
    try:
        cash      = bs.loc["Cash And Cash Equivalents"].iloc[0]
        total_debt= bs.loc["Total Debt"].iloc[0]
        net_debt  = total_debt - cash
    except:
        net_debt = 0

    shares = info.get("sharesOutstanding", 1)
    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

    # Project FCFs
    proj_fcfs = [fcf * (1+g_rate)**yr for yr in range(1, int(years)+1)]
    terminal  = proj_fcfs[-1] * (1+term_g) / (wacc - term_g) if wacc > term_g else proj_fcfs[-1]*15
    disc_fcfs = [f/(1+wacc)**yr for yr,f in enumerate(proj_fcfs,1)]
    disc_term = terminal/(1+wacc)**int(years)
    enterprise_value = sum(disc_fcfs) + disc_term
    equity_value     = enterprise_value - net_debt
    intrinsic_per_share = equity_value / shares if shares else 0
    margin_of_safety = (intrinsic_per_share - current_price) / intrinsic_per_share * 100 if intrinsic_per_share else 0

    val_color = "#4caf7d" if intrinsic_per_share > current_price else "#e05c5c"
    mos_color = "#4caf7d" if margin_of_safety > 20 else ("#c8953a" if margin_of_safety > 0 else "#e05c5c")

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Intrinsic Value",f"${intrinsic_per_share:.2f}","DCF estimate per share",val_color)
    with c2: metric_card("Current Price",f"${current_price:.2f}" if current_price else "—","market price","#c8953a")
    with c3: metric_card("Margin of Safety",f"{margin_of_safety:+.1f}%","positive = undervalued",mos_color)
    with c4: metric_card("Base FCF",f"${fcf/1e9:.2f}B" if fcf>=1e9 else f"${fcf/1e6:.0f}M","trailing free cash flow","#6b6e6c")

    st.markdown("<br>",unsafe_allow_html=True)

    # Waterfall chart
    labels  = [f"Year {i}" for i in range(1,int(years)+1)] + ["Terminal Value","Intrinsic / Share"]
    values  = disc_fcfs + [disc_term, intrinsic_per_share/1e9 if intrinsic_per_share>1e9 else intrinsic_per_share/1e6]
    bar_col = ["#5bc8c8"]*int(years) + ["#c8953a","#4caf7d" if intrinsic_per_share>current_price else "#e05c5c"]

    fig = go.Figure()
    # Projection bars
    fig.add_trace(go.Bar(
        x=[f"Year {i}" for i in range(1,int(years)+1)],
        y=[f/1e9 for f in disc_fcfs],
        name="Discounted FCF",marker_color="#5bc8c8",opacity=0.8))
    fig.add_trace(go.Bar(x=["Terminal Value"],y=[disc_term/1e9],name="Terminal Value",marker_color="#c8953a",opacity=0.8))
    _mkt_y = current_price/1e9 if current_price>1e6 else current_price/1e6
    fig.add_hline(y=_mkt_y,line_color="#e8c97a",line_dash="dash",line_width=1.5)
    fig.add_annotation(x=1.01,y=_mkt_y,xref="paper",yref="y",
        text=f"Market ${current_price:.2f}",showarrow=False,xanchor="left",
        font=dict(color="#e8c97a",size=9))
    layout=CHART_LAYOUT.copy()
    layout.update(height=300,margin=dict(l=0,r=80,t=24,b=0),
        yaxis=dict(showgrid=True,gridcolor="#1f2020",color="#6b6e6c",
                   title="$B" if fcf>=1e9 else "$M",tickfont=dict(size=10)),
        title=dict(text=f"{ticker} — DCF Cash Flow Projection",
                   font=dict(size=13,color="#e8e4dc",family="Fraunces, serif"),x=0),
        barmode="group",bargap=0.2)
    fig.update_layout(**layout)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    verdict = ("undervalued" if margin_of_safety > 20 else "fairly valued" if margin_of_safety > 0 else "overvalued")
    v_color = "#4caf7d" if margin_of_safety>20 else("#c8953a" if margin_of_safety>0 else "#e05c5c")
    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid {v_color};
        padding:16px 20px;border-radius:3px;">
        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:{v_color};margin-bottom:6px;">DCF Verdict</div>
        <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">
            At a {wacc*100:.1f}% WACC and {g_rate*100:.1f}% growth rate, <b style="color:#e8e4dc;">{ticker}</b>
            appears <b style="color:{v_color};">{verdict}</b> with a {margin_of_safety:+.1f}% margin of safety.<br><br>
            <b style="color:#e8e4dc;">Important:</b> DCF is highly sensitive to the growth and WACC assumptions.
            Use the sliders above to run bull/base/bear scenarios. A margin of safety above 20% is typically
            required before considering a position.
        </div>
    </div>""",unsafe_allow_html=True)

# ── 7. CAN SLIM ───────────────────────────────────────────
def model_canslim():
    info_card("CAN SLIM Ranking",
        "William O'Neil's growth stock scoring system. Each letter scores one criterion: "
        "Current earnings, Annual earnings, New highs, Supply/demand, Leader vs laggard, "
        "Institutional sponsorship, Market direction. Scores 5/7 or higher signal strong candidates.")

    ticker, run = model_ticker_input("canslim")
    if not run or not ticker: return

    with st.spinner(f"Running CAN SLIM analysis for {ticker}…"):
        t    = yf.Ticker(ticker)
        info = get_info(ticker)
        hist = get_hist(ticker)
        hist_spy = get_hist("SPY")

    results = {}

    # C — Current quarterly EPS growth > 25%
    # .quarterly_earnings deprecated in yfinance 0.2.x — use quarterly_income_stmt
    try:
        qi = t.quarterly_income_stmt
        eps_row = None
        for _row in ["Basic EPS","Diluted EPS","Net Income"]:
            if _row in qi.index: eps_row = qi.loc[_row]; break
        if eps_row is not None and len(eps_row)>=2:
            cur  = eps_row.iloc[0]
            prev = eps_row.iloc[4] if len(eps_row)>=5 else eps_row.iloc[1]
            c_growth = (cur-prev)/abs(prev)*100 if prev and prev!=0 else 0
            results["C — Current EPS Growth"] = (min(100,max(0,c_growth)), f"{c_growth:+.1f}% YoY quarterly EPS", c_growth>=25)
        else:
            results["C — Current EPS Growth"] = (0,"Quarterly income statement unavailable",False)
    except:
        results["C — Current EPS Growth"] = (0,"Could not retrieve quarterly EPS",False)

    # A — Annual EPS growth > 25% sustained
    # .earnings deprecated in yfinance 0.2.x — use income_stmt
    try:
        ai = t.income_stmt
        eps_row_a = None
        for _row in ["Basic EPS","Diluted EPS","Net Income"]:
            if _row in ai.index: eps_row_a = ai.loc[_row]; break
        if eps_row_a is not None and len(eps_row_a)>=2:
            a_growth = (eps_row_a.iloc[0]-eps_row_a.iloc[1])/abs(eps_row_a.iloc[1])*100 if eps_row_a.iloc[1]!=0 else 0
            results["A — Annual EPS Growth"] = (min(100,max(0,a_growth)), f"{a_growth:+.1f}% annual EPS growth", a_growth>=25)
        else:
            results["A — Annual EPS Growth"] = (0,"Annual income statement unavailable",False)
    except:
        results["A — Annual EPS Growth"] = (0,"Could not retrieve annual EPS",False)

    # N — New 52-week highs proximity
    try:
        if hist is not None:
            high52 = hist["Close"].max(); current = hist["Close"].iloc[-1]
            pct_from_high = (current/high52)*100
            passes = pct_from_high >= 85
            results["N — Near New Highs"] = (pct_from_high, f"{pct_from_high:.1f}% of 52-week high (${high52:.2f})", passes)
        else: results["N — Near New Highs"] = (0,"No price data",False)
    except: results["N — Near New Highs"] = (0,"Could not calculate",False)

    # S — Supply/demand (up-volume vs down-volume ratio)
    try:
        if hist is not None:
            c_close = hist["Close"]; vol = hist["Volume"]
            up_vol = vol[c_close.diff()>0].sum(); dn_vol = vol[c_close.diff()<=0].sum()
            ratio = up_vol/(dn_vol if dn_vol>0 else 1)
            passes = ratio > 1.2
            results["S — Supply/Demand (Vol)"] = (min(100,ratio*50), f"Up/Down volume ratio: {ratio:.2f}", passes)
        else: results["S — Supply/Demand (Vol)"] = (0,"No volume data",False)
    except: results["S — Supply/Demand (Vol)"] = (0,"Could not calculate",False)

    # L — Leader vs Laggard (vs S&P over 6 months)
    try:
        if hist is not None and hist_spy is not None:
            common = hist.index.intersection(hist_spy.index)
            n = min(126,len(common))
            stock_ret = hist.loc[common,"Close"].iloc[-1]/hist.loc[common,"Close"].iloc[-n]-1
            spy_ret   = hist_spy.loc[common,"Close"].iloc[-1]/hist_spy.loc[common,"Close"].iloc[-n]-1
            rel = (stock_ret-spy_ret)*100
            passes = rel > 0
            results["L — Leader vs Laggard"] = (min(100,max(0,50+rel)), f"{rel:+.1f}% vs SPY over 6 months", passes)
        else: results["L — Leader vs Laggard"] = (0,"No comparison data",False)
    except: results["L — Leader vs Laggard"] = (0,"Could not calculate",False)

    # I — Institutional ownership via yfinance
    # Sweet spot: 20–80% — some coverage = legitimacy, too much = crowded/no room to run
    try:
        inst_pct       = (info.get("heldPercentInstitutions") or 0) * 100
        inst_count     = info.get("institutionsCount") or info.get("institutionCount") or 0
        passes         = 20 <= inst_pct <= 80
        inst_desc      = f"{inst_pct:.1f}% held by institutions"
        if inst_count:  inst_desc += f" across {inst_count:,} funds"
        if inst_pct < 20:   inst_desc += " — too little coverage (unproven)"
        elif inst_pct > 80: inst_desc += " — heavily owned (limited upside fuel)"
        else:               inst_desc += " — healthy sponsorship level"
        results["I — Institutional Ownership"] = (inst_pct, inst_desc, passes)
    except:
        results["I — Institutional Ownership"] = (0, "Could not retrieve ownership data", False)

    # M — Market direction (SPY above 50-day MA)
    try:
        if hist_spy is not None:
            spy_c  = hist_spy["Close"]
            spy_ma = sma(spy_c, 50).iloc[-1]
            passes = spy_c.iloc[-1] > spy_ma
            results["M — Market Direction"] = (100 if passes else 0,
                f"SPY {'above' if passes else 'below'} 50-day MA (${spy_ma:.2f})", passes)
        else: results["M — Market Direction"] = (0,"No SPY data",False)
    except: results["M — Market Direction"] = (0,"Could not calculate",False)

    # Score
    passed = sum(1 for _,(score,desc,p) in results.items() if p)
    total_score = passed / 7 * 100

    s_color = "#4caf7d" if passed>=5 else("#c8953a" if passed>=4 else "#e05c5c")
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("CAN SLIM Score", f"{passed}/7", "5+ = strong candidate", s_color)
    with c2: metric_card("Rating", "Strong" if passed>=5 else("Moderate" if passed>=4 else "Weak"),
                         "by O'Neil's methodology", s_color)
    with c3: metric_card("Market Condition", "Confirmed Uptrend" if results["M — Market Direction"][2] else "Under Pressure",
                         "M criterion", "#4caf7d" if results["M — Market Direction"][2] else "#e05c5c")

    st.markdown("<br>",unsafe_allow_html=True)
    section_header("Criterion Breakdown")

    for criterion,(score,desc,passes) in results.items():
        letter = criterion.split(" — ")[0]
        label  = criterion.split(" — ")[1]
        p_color= "#4caf7d" if passes else "#e05c5c"
        p_icon = "✓" if passes else "✗"
        st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid {p_color};
            padding:12px 16px;border-radius:3px;margin-bottom:6px;display:flex;align-items:center;">
            <span style="font-family:'Fraunces',serif;font-size:1.2rem;color:{p_color};min-width:32px;">{p_icon}</span>
            <div>
                <div style="font-size:10px;color:{p_color};letter-spacing:.1em;text-transform:uppercase;">{letter} &nbsp;·&nbsp; {label}</div>
                <div style="font-size:11px;color:#a0a8a4;margin-top:3px;">{desc}</div>
            </div>
        </div>""",unsafe_allow_html=True)

    # Radar chart
    criteria_labels = [c.split(" — ")[0] for c in results.keys()]
    criteria_values = [min(100,s) for s,_,_ in results.values()]
    criteria_values += [criteria_values[0]]
    criteria_labels += [criteria_labels[0]]

    col_radar, col_holders = st.columns([1, 1])

    with col_radar:
        fig = go.Figure(go.Scatterpolar(r=criteria_values,theta=criteria_labels,fill="toself",
            fillcolor="rgba(200,149,58,0.13)",line=dict(color="#c8953a",width=2),name=ticker))
        fig.update_layout(height=320,margin=dict(l=20,r=20,t=20,b=20),paper_bgcolor="#111210",
            polar=dict(bgcolor="#181917",
                       radialaxis=dict(visible=True,range=[0,100],color="#6b6e6c",gridcolor="#2a2c2b",tickfont=dict(size=8)),
                       angularaxis=dict(color="#a0a8a4",gridcolor="#2a2c2b",tickfont=dict(size=11))),
            font=dict(color="#a0a8a4",family="DM Mono"))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with col_holders:
        section_header("Top Institutional Holders", "via yfinance 13F data")
        try:
            holders_df = yf.Ticker(ticker).institutional_holders
            if holders_df is not None and len(holders_df) > 0:
                # Clean up columns
                display_cols = [c for c in ["Holder","Shares","% Out","Value"] if c in holders_df.columns]
                holders_display = holders_df[display_cols].head(8).copy()
                # Format shares and value
                if "Shares" in holders_display.columns:
                    holders_display["Shares"] = holders_display["Shares"].apply(
                        lambda x: f"{x/1e6:.1f}M" if x>=1e6 else f"{x/1e3:.0f}K")
                if "Value" in holders_display.columns:
                    holders_display["Value"] = holders_display["Value"].apply(
                        lambda x: f"${x/1e9:.1f}B" if x>=1e9 else f"${x/1e6:.0f}M")
                if "% Out" in holders_display.columns:
                    holders_display["% Out"] = holders_display["% Out"].apply(
                        lambda x: f"{x*100:.2f}%" if x < 1 else f"{x:.2f}%")
                st.dataframe(
                    holders_display.style.set_properties(**{
                        "background-color":"#111210","color":"#e8e4dc",
                        "border-color":"#1f2020","font-family":"DM Mono, monospace","font-size":"11px"
                    }).set_table_styles([
                        {"selector":"th","props":[("background-color","#0b0a09"),("color","#6b6e6c"),
                         ("font-size","9px"),("letter-spacing","0.15em"),("text-transform","uppercase"),
                         ("border-color","#1f2020"),("padding","8px 10px")]},
                        {"selector":"td","props":[("padding","8px 10px"),("border-color","#1f2020")]},
                    ]),
                    use_container_width=True, hide_index=True
                )
                # Insider ownership note
                insider_pct = (info.get("heldPercentInsiders") or 0) * 100
                if insider_pct > 0:
                    ic = "#4caf7d" if insider_pct >= 5 else "#6b6e6c"
                    st.markdown(f"""<div style="font-size:10px;color:{ic};margin-top:8px;">
                        Insider ownership: {insider_pct:.1f}%
                        {"— significant skin in the game ✓" if insider_pct>=5 else "— low insider stake"}</div>""",
                        unsafe_allow_html=True)
            else:
                st.markdown("""<div style="font-size:12px;color:#6b6e6c;padding-top:12px;">
                    No institutional holder data available for this ticker.</div>""",unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""<div style="font-size:12px;color:#6b6e6c;padding-top:12px;">
                Could not load holder data.</div>""",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  LEARN PAGE
# ─────────────────────────────────────────────────────────
INDICATOR_DOCS = {
    "Moving Averages":[
        {"name":"Simple Moving Average (SMA)","variants":"SMA 10, 20, 30, 50, 100, 200","difficulty":"Beginner",
         "what":"The plain average of closing prices over N days. Every day gets equal weight.",
         "how":"Price above SMA = bullish trend. Below = bearish. Longer SMAs (200) show the big picture; shorter ones (10) react quickly.",
         "buy":"Price crosses above the SMA","sell":"Price crosses below the SMA",
         "note":"The SMA 200 is one of Wall Street's most watched levels. Institutional funds often use it as a key support/resistance line."},
        {"name":"Exponential Moving Average (EMA)","variants":"EMA 10, 20, 30, 50, 100, 200","difficulty":"Beginner",
         "what":"Like SMA but recent prices carry more weight, making it faster to react to new information.",
         "how":"Signals trend changes earlier than SMA but also generates more false signals. Used for timing vs trend confirmation.",
         "buy":"Price above EMA and EMA sloping upward","sell":"Price drops below EMA or EMA slopes downward",
         "note":"The Golden Cross (EMA 50 crossing above EMA 200) is one of the most famous bullish signals in technical analysis."},
        {"name":"Hull Moving Average (HMA 9)","variants":"HMA 9","difficulty":"Intermediate",
         "what":"Designed to reduce lag without sacrificing smoothness using weighted MAs in a two-step formula.",
         "how":"Far more responsive than SMA or EMA. Popular for short-term trading where current momentum matters most.",
         "buy":"HMA slopes upward and price is above it","sell":"HMA slopes downward and price is below it",
         "note":"The square root step in HMA's formula gives it its unique lag-reduction property."},
        {"name":"Volume Weighted Moving Average (VWMA 20)","variants":"VWMA 20","difficulty":"Intermediate",
         "what":"Like SMA but each price is weighted by that day's volume — high-volume days matter more.",
         "how":"Reflects the true average price investors paid. Deviations between VWMA and SMA show whether price moves had high or low conviction.",
         "buy":"Price is above VWMA (demand is strong)","sell":"Price falls below VWMA (sellers dominating)",
         "note":"Institutional traders reference VWAP (intraday version) as a benchmark for executing large orders."},
        {"name":"Ichimoku Base Line","variants":"Ichimoku (9, 26, 52, 26)","difficulty":"Advanced",
         "what":"Part of the Ichimoku Cloud system. The Base Line is the midpoint of the 26-period high-low range — the equilibrium price.",
         "how":"Price above the Base Line = medium-term bullish trend. We use it as a standalone trend filter.",
         "buy":"Price is above the Base Line","sell":"Price is below the Base Line",
         "note":"Developed by Goichi Hosoda in the 1930s. The name means 'one glance equilibrium chart' in Japanese."},
    ],
    "Oscillators":[
        {"name":"RSI","variants":"RSI 14","difficulty":"Beginner",
         "what":"Measures speed and size of recent price changes on a 0–100 scale. Reveals if a stock is overbought or oversold.",
         "how":"Standard thresholds: 30 = oversold, 70 = overbought. Torosian adjusts these by sector.",
         "buy":"RSI drops below sector-adjusted oversold level","sell":"RSI rises above sector-adjusted overbought level",
         "note":"Developed by J. Welles Wilder in 1978. Sector adjustment separates professional from amateur analysis."},
        {"name":"MACD","variants":"MACD (12, 26)","difficulty":"Beginner",
         "what":"Measures the relationship between two EMAs. When the faster EMA diverges from the slower, it signals momentum.",
         "how":"MACD Line = EMA(12) minus EMA(26). Signal Line = 9-day EMA of MACD. Crossovers generate signals.",
         "buy":"MACD Line crosses above the Signal Line","sell":"MACD Line crosses below the Signal Line",
         "note":"Developed by Gerald Appel in the 1970s. The 12-26-9 parameters are the universal standard."},
        {"name":"Stochastic %K","variants":"Stoch %K (14, 3, 3)","difficulty":"Beginner",
         "what":"Compares closing price to the 14-day high-low range on 0–100. Shows where price sits within its recent range.",
         "how":"Near 100 = closing near period highs. Near 0 = closing near lows. Extremes signal reversals.",
         "buy":"Stochastic drops below 20","sell":"Stochastic rises above 80",
         "note":"Developed by George Lane in the 1950s. Smoothing reduces noise significantly."},
        {"name":"Stochastic RSI","variants":"Stoch RSI (3, 3, 14, 14)","difficulty":"Intermediate",
         "what":"Applies Stochastic formula to RSI values — an indicator of an indicator, more sensitive to short-term shifts.",
         "how":"Ranges 0–1. Below 0.2 = RSI near its recent low. Above 0.8 = RSI near its recent high.",
         "buy":"Stoch RSI drops below 0.2","sell":"Stoch RSI rises above 0.8",
         "note":"More volatile than RSI — best used alongside slower indicators for confirmation."},
        {"name":"CCI","variants":"CCI (20)","difficulty":"Intermediate",
         "what":"Measures how far price is from its 20-day average, normalised by typical volatility.",
         "how":"Readings beyond ±100 are significant — the stock has moved unusually far from its average.",
         "buy":"CCI drops below sector-adjusted level (~-100)","sell":"CCI rises above sector-adjusted level (~+100)",
         "note":"Developed by Donald Lambert in 1980. Has no fixed ceiling or floor — ±200 readings do occur."},
        {"name":"Williams %R","variants":"Williams %R (14)","difficulty":"Beginner",
         "what":"Nearly identical to Stochastic but inverted. Measures close vs 14-day range on -100 to 0 scale.",
         "how":"Near 0 = closing near highs (strong). Near -100 = closing near lows (weak).",
         "buy":"W%R drops below sector-adjusted level (~-80)","sell":"W%R rises above sector-adjusted level (~-20)",
         "note":"The negative scale confuses beginners: -10 is overbought and -90 is oversold."},
        {"name":"ADX","variants":"ADX (14)","difficulty":"Intermediate",
         "what":"Measures trend strength, not direction. Combines +DI and -DI to show how powerful the trend is.",
         "how":"ADX > 25 = strong trend. +DI vs -DI determines direction.",
         "buy":"ADX > 25 and +DI above -DI","sell":"ADX > 25 and -DI above +DI",
         "note":"ADX below 20 = ranging market. Other signals become less reliable in this condition."},
        {"name":"Awesome Oscillator","variants":"Awesome Oscillator","difficulty":"Beginner",
         "what":"Measures momentum by comparing 5-period and 34-period SMA of the midpoint price.",
         "how":"Positive = bullish momentum. Negative = bearish. No overbought/oversold levels.",
         "buy":"Crosses above zero","sell":"Crosses below zero",
         "note":"Created by Bill Williams. Simple design reduces over-optimisation risk."},
        {"name":"Momentum (10)","variants":"Momentum (10)","difficulty":"Beginner",
         "what":"The simplest momentum indicator: today's price minus price 10 days ago.",
         "how":"Positive and rising = trend accelerating. Positive but falling = trend losing steam.",
         "buy":"Positive (price higher than 10 days ago)","sell":"Negative (price lower than 10 days ago)",
         "note":"In economics terms, this is a first-difference of price — like measuring GDP growth rate."},
        {"name":"Bull Bear Power","variants":"Bull Bear Power","difficulty":"Intermediate",
         "what":"Distance between high/low prices and a 13-period EMA. Bull Power = High minus EMA, Bear Power = Low minus EMA.",
         "how":"Positive combined value = bulls in control. Negative = bears in control.",
         "buy":"Combined Bull+Bear Power is positive","sell":"Combined Bull+Bear Power is negative",
         "note":"Developed by Dr. Alexander Elder. Best paired with a trend filter."},
        {"name":"Ultimate Oscillator","variants":"Ult. Osc. (7, 14, 28)","difficulty":"Advanced",
         "what":"Combines three timeframes (7, 14, 28) into one oscillator with weighted averaging to reduce false signals.",
         "how":"Ranges 0–100. Shorter period weighted 4x, medium 2x, long 1x.",
         "buy":"Drops below 30","sell":"Rises above 70",
         "note":"Also by Larry Williams. Multi-timeframe approach solves single-period sensitivity issues."},
    ]
}
DIFF_COLOR={"Beginner":"#4caf7d","Intermediate":"#c8953a","Advanced":"#e05c5c"}

def render_learn():
    st.markdown("""
    <div style="padding:52px 0 40px;text-align:center;border-bottom:1px solid #1f2020;margin-bottom:40px;">
        <div style="font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:#c8953a;margin-bottom:14px;">Torosian Stock Insights</div>
        <h1 style="font-family:'Fraunces',serif;font-size:clamp(1.8rem,5vw,3rem);font-weight:300;color:#e8e4dc;margin:0 0 14px;line-height:1.1;">
            Understand every indicator.<br><em style="color:#c8953a;">Make better decisions.</em></h1>
        <p style="font-size:13px;color:#6b6e6c;max-width:520px;margin:0 auto;line-height:1.8;">
            A plain-English guide to all 26 technical indicators used in this platform.</p>
    </div>""",unsafe_allow_html=True)

    # Sector thresholds
    rows=[[s,f"RSI: {t['rsi_os']}/{t['rsi_ob']}",f"CCI: {t['cci_os']}/{t['cci_ob']}",f"W%R: {t['wpr_os']}/{t['wpr_ob']}"]
          for s,t in SECTOR_THRESHOLDS.items()]
    df_thresh=pd.DataFrame(rows,columns=["Sector","RSI (OS/OB)","CCI (OS/OB)","W%R (OS/OB)"])
    st.markdown("""<div style="background:#181917;border:1px solid #2a2c2b;border-left:3px solid #c8953a;
        padding:18px 22px;border-radius:3px;margin-bottom:12px;">
        <div style="font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:#c8953a;margin-bottom:8px;">Sector-Adjusted Thresholds</div>
        <p style="font-size:12px;color:#a0a8a4;line-height:1.8;margin:0 0 14px;">
            Torosian adjusts RSI, CCI, and Williams %R thresholds by sector — because a Technology stock at RSI 71 is normal,
            while a Real Estate stock at RSI 66 is already overbought.</p>
    </div>""",unsafe_allow_html=True)
    st.dataframe(style_df(df_thresh),use_container_width=True,hide_index=True)

    st.markdown("<br>",unsafe_allow_html=True)
    for group,inds in INDICATOR_DOCS.items():
        st.markdown(f"""<div style="font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:#6b6e6c;
            margin-bottom:18px;padding-top:14px;border-top:1px solid #1f2020;">{group}</div>""",unsafe_allow_html=True)
        for ind in inds:
            dc=DIFF_COLOR.get(ind["difficulty"],"#6b6e6c")
            with st.expander(f"  {ind['name']}  —  {ind['variants']}"):
                st.markdown(f"""
                <div style="margin-bottom:12px;"><span style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;
                    background:{dc}22;color:{dc};padding:3px 8px;border-radius:2px;">{ind['difficulty']}</span></div>
                <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#6b6e6c;margin-bottom:5px;">What it measures</div>
                <p style="font-size:12px;color:#a0a8a4;line-height:1.8;margin-bottom:12px;">{ind['what']}</p>
                <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#6b6e6c;margin-bottom:5px;">How to read it</div>
                <p style="font-size:12px;color:#a0a8a4;line-height:1.8;margin-bottom:12px;">{ind['how']}</p>
                <div style="display:flex;gap:10px;margin-bottom:12px;">
                    <div style="background:rgba(76,175,125,0.07);border:1px solid rgba(76,175,125,0.2);padding:10px 14px;border-radius:3px;flex:1;">
                        <div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#4caf7d;margin-bottom:3px;">Buy signal</div>
                        <div style="font-size:11px;color:#a0a8a4;">{ind['buy']}</div>
                    </div>
                    <div style="background:rgba(224,92,92,0.07);border:1px solid rgba(224,92,92,0.2);padding:10px 14px;border-radius:3px;flex:1;">
                        <div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#e05c5c;margin-bottom:3px;">Sell signal</div>
                        <div style="font-size:11px;color:#a0a8a4;">{ind['sell']}</div>
                    </div>
                </div>
                <div style="background:#111210;border:1px solid #1f2020;padding:10px 14px;border-radius:3px;">
                    <span style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#c8953a;">Did you know · </span>
                    <span style="font-size:11px;color:#6b6e6c;">{ind['note']}</span>
                </div>""",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  SCREENER APP PAGE
# ─────────────────────────────────────────────────────────
def render_screener_sidebar():
    """Returns all screener filter values from sidebar widgets."""
    st.markdown("""<div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;margin-bottom:10px;">Core Filters</div>""",unsafe_allow_html=True)
    all_sectors=sorted(set(s for _,_,s,_ in STOCK_UNIVERSE))
    sector=st.selectbox("Sector",["ALL"]+all_sectors)
    cap=st.selectbox("Market Cap",["All","Large (>$10B)","Mid ($2B–$10B)","Small (<$2B)"])
    cap_filter={"All":"All","Large (>$10B)":"Large","Mid ($2B–$10B)":"Mid","Small (<$2B)":"Small"}[cap]
    # Note: S&P 500 members are all Large cap — Mid/Small filter will return 0 results
    risk_key=st.selectbox("Risk Tolerance",list(RISK_RANGES.keys()))
    beta_min,beta_max=RISK_RANGES[risk_key]
    horizon=st.radio("Horizon",["Short Term","Long Term"])
    min_score=st.slider("Min Technical Score",0,90,0,5)
    top_n=st.slider("Results to Show",3,20,8)

    use_ind_filter=False; req_indicators=[]; req_direction="Bullish"
    sort_mode="Composite Score"; sort_indicator=None; sort_dir="Most Bullish First"
    st.markdown("<br>",unsafe_allow_html=True)
    with st.expander("⚙  Advanced Filters"):
        st.markdown("""<div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#c8953a;margin-bottom:8px;">Indicator Filter</div>""",unsafe_allow_html=True)
        use_ind_filter=st.checkbox("Filter by specific indicators")
        if use_ind_filter:
            req_indicators=st.multiselect("Required indicators",ALL_INDICATORS)
            req_direction=st.radio("Signal direction",["Bullish","Bearish"])
        st.markdown("""<div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#c8953a;margin:12px 0 8px;">Sort Mode</div>""",unsafe_allow_html=True)
        sort_mode=st.radio("Sort results by",["Composite Score","Single Indicator"])
        if sort_mode=="Single Indicator":
            sort_indicator=st.selectbox("Choose indicator",ALL_INDICATORS)
            sort_dir=st.radio("Rank direction",["Most Bullish First","Most Bearish First"])

    st.markdown("<br>",unsafe_allow_html=True)
    run=st.button("Run Screener →",use_container_width=True)
    return (sector,cap_filter,beta_min,beta_max,horizon,min_score,top_n,
            use_ind_filter,req_indicators,req_direction,sort_mode,sort_indicator,sort_dir,run)

def render_app():
    (sector,cap_filter,beta_min,beta_max,horizon,min_score,top_n,
     use_ind_filter,req_indicators,req_direction,sort_mode,sort_indicator,sort_dir,run) = render_screener_sidebar()

    # Header
    st.markdown("""<div style="padding:24px 0 18px;border-bottom:1px solid #1f2020;margin-bottom:24px;">
        <div style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#c8953a;margin-bottom:5px;">Torosian Stock Insights</div>
        <h1 style="font-family:'Fraunces',serif;font-size:1.9rem;font-weight:300;color:#e8e4dc;margin:0;">
            Your personal <em style="color:#c8953a;">equity research</em> dashboard</h1>
        <p style="color:#6b6e6c;font-size:11px;margin-top:6px;">26 indicators · Sector-adjusted thresholds · Live from Yahoo Finance</p>
    </div>""",unsafe_allow_html=True)

    tab_screen,tab_deep,tab_compare=st.tabs(["📋  Screener","🔍  Deep Dive","⚔  Compare"])

    # ── SCREENER ─────────────────────────────────────────
    with tab_screen:
        # Show universe status
        if st.session_state.get("universe_fallback"):
            st.markdown("""<div style="background:#1a1512;border:1px solid #c8953a44;border-left:3px solid #c8953a;
                padding:10px 16px;border-radius:3px;margin-bottom:12px;font-size:11px;color:#c8953a;">
                ⚠ Wikipedia fetch failed — running on 52-stock fallback universe. Live data will resume automatically.</div>""",
                unsafe_allow_html=True)
        else:
            n_stocks = len(st.session_state.get("stock_universe",[]))
            st.markdown(f"""<div style="background:#101810;border:1px solid #4caf7d44;border-left:3px solid #4caf7d;
                padding:10px 16px;border-radius:3px;margin-bottom:12px;font-size:11px;color:#4caf7d;">
                ✓ Live S&P 500 universe loaded — {n_stocks} stocks · refreshes every 24 hours</div>""",
                unsafe_allow_html=True)

        if not run:
            st.markdown("""<div style="text-align:center;padding:60px 0;">
                <div style="font-size:3rem;margin-bottom:14px;">📈</div>
                <div style="font-family:'Fraunces',serif;font-size:1.2rem;color:#6b6e6c;font-weight:300;">
                    Configure your filters in the sidebar and press <em>Run Screener</em></div>
            </div>""",unsafe_allow_html=True)
        else:
            candidates=[(t,n,s,c) for t,n,s,c in STOCK_UNIVERSE
                        if (sector=="ALL" or s==sector) and (cap_filter=="All" or c==cap_filter)]
            if not candidates: st.error("No stocks match your filters."); st.stop()
            results=[]; prog=st.progress(0,text="Fetching live data…")
            for i,(ticker,name,sec,cap_size) in enumerate(candidates):
                prog.progress((i+1)/len(candidates),text=f"Analysing {ticker}…")
                info=get_info(ticker)
                beta = info.get("beta")
                # If beta is missing, assume market beta of 1.0 so stock isn't silently dropped
                if beta is None:
                    beta = 1.0
                if not (beta_min <= beta < beta_max): continue
                hist=get_hist(ticker)
                try:
                    score,signals=compute_score(hist,sec)
                except Exception:
                    score,signals=None,{}
                if score is None or score<min_score: continue
                if use_ind_filter and req_indicators:
                    rv=1 if req_direction=="Bullish" else -1
                    if not all(signals.get(ind,0)==rv for ind in req_indicators): continue
                price=info.get("currentPrice") or info.get("regularMarketPrice")
                pe=info.get("trailingPE"); div=(info.get("dividendYield") or 0)*100
                mcap=info.get("marketCap"); target=info.get("targetMeanPrice")
                upside=((target-price)/price*100) if price and target else None
                results.append({"ticker":ticker,"name":name,"sector":sec,"cap":cap_size,
                    "price":price,"beta":beta,"pe":pe,"div":div,"mcap":mcap,"target":target,
                    "upside":upside,"rec":analyst_label(info.get("recommendationKey","")),
                    "n_analysts":info.get("numberOfAnalystOpinions"),
                    "score":score,"signals":signals,"hist":hist,"info":info})
            prog.empty()
            if not results:
                st.error(
                    f"No stocks passed all filters — {len(candidates)} candidates checked. "
                    "Try: set Risk Tolerance to 'Medium', Min Score to 0, and Sector to ALL."
                )
                st.stop()

            if sort_mode=="Single Indicator" and sort_indicator:
                rev=(sort_dir=="Most Bullish First")
                results.sort(key=lambda x:x["signals"].get(sort_indicator,0),reverse=rev)
                sort_note=f"Sorted by {sort_indicator}"
            elif horizon=="Long Term":
                results.sort(key=lambda x:(x["score"],x["div"]),reverse=True); sort_note="Score + Yield"
            else:
                results.sort(key=lambda x:x["score"],reverse=True); sort_note="Composite Score"
            top=results[:top_n]

            avg_sc=round(sum(r["score"] for r in top)/len(top),1)
            buys=sum(1 for r in top if r["score"]>=55); sells=sum(1 for r in top if r["score"]<45)
            c1,c2,c3,c4=st.columns(4)
            with c1: metric_card("Stocks Found",str(len(top)),f"of {len(results)} passed","#c8953a")
            with c2: metric_card("Avg Score",f"{avg_sc}/100",score_label(avg_sc),score_color(avg_sc))
            with c3: metric_card("Buy Signals",str(buys),f"{len(top)-buys-sells} neutral · {sells} sell","#4caf7d")
            with c4: metric_card("Sort Mode",sort_note,horizon,"#5bc8c8")

            if use_ind_filter and req_indicators:
                badges=" &nbsp;".join([f'<span style="background:rgba(200,149,58,0.13);color:#c8953a;font-size:10px;padding:2px 8px;border-radius:2px;">{i}</span>' for i in req_indicators])
                st.markdown(f'<div style="margin:10px 0 4px;font-size:11px;color:#6b6e6c;">Filtered: {req_direction} on &nbsp;{badges}</div>',unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)
            rows=[{"Ticker":r["ticker"],"Company":r["name"],"Cap":r["cap"],
                   "Price":f"${r['price']:.2f}" if r["price"] else "—",
                   "Beta":f"{r['beta']:.2f}","P/E":f"{r['pe']:.1f}" if r["pe"] else "—",
                   "Div %":f"{r['div']:.2f}%","Upside":f"{r['upside']:+.1f}%" if r["upside"] else "—",
                   "Consensus":r["rec"],"Score":r["score"],"Signal":score_label(r["score"])} for r in top]
            st.dataframe(style_df(pd.DataFrame(rows)),use_container_width=True,hide_index=True,height=380)

            top_r=top[0]; th=get_thresh(top_r["sector"])
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:#6b6e6c;margin-bottom:10px;">
                Signal Breakdown — {top_r['ticker']}
                <span style="color:#c8953a"> · {top_r['sector']} thresholds: RSI {th['rsi_os']}/{th['rsi_ob']}</span></div>""",unsafe_allow_html=True)
            col_chart,col_sum=st.columns([2,1])
            with col_chart: st.plotly_chart(signals_chart(top_r["signals"]),use_container_width=True,config={"displayModeBar":False})
            with col_sum:
                sigs=top_r["signals"]
                ma_b=sum(1 for n in MA_NAMES if sigs.get(n,0)==1); ma_s=sum(1 for n in MA_NAMES if sigs.get(n,0)==-1)
                osc_b=sum(1 for n in OSC_NAMES if sigs.get(n,0)==1); osc_s=sum(1 for n in OSC_NAMES if sigs.get(n,0)==-1)
                tb=ma_b+osc_b; ts=ma_s+osc_s; pct=tb/26*100
                interp=("Strong bullish consensus" if pct>=65 else "Mildly bullish" if pct>=50 else "Bearish lean" if ts>tb else "Mixed signals")
                ic=("#4caf7d" if pct>=65 else "#8bc98b" if pct>=50 else "#e05c5c" if ts>tb else "#5bc8c8")
                st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;padding:18px;border-radius:3px;">
                    <div style="font-family:'Fraunces',serif;font-size:1rem;color:{ic};margin-bottom:10px;">{interp}</div>
                    <div style="font-size:11px;color:#a0a8a4;line-height:2.1;">
                        <span style="color:#4caf7d">▲ {tb} Buy</span> &nbsp;<span style="color:#e05c5c">▼ {ts} Sell</span>
                        &nbsp;<span style="color:#6b6e6c">● {26-tb-ts} N</span><br>
                        MAs: <span style="color:#4caf7d">{ma_b}↑</span>/<span style="color:#e05c5c">{ma_s}↓</span>
                        &nbsp;Oscs: <span style="color:#4caf7d">{osc_b}↑</span>/<span style="color:#e05c5c">{osc_s}↓</span>
                    </div>
                    <hr style="border-color:#2a2c2b;margin:10px 0;">
                    <div style="font-size:9px;color:#6b6e6c;line-height:2;">RSI: {th['rsi_os']}/{th['rsi_ob']}<br>CCI: {th['cci_os']}/{th['cci_ob']}<br>W%R: {th['wpr_os']}/{th['wpr_ob']}</div>
                    <hr style="border-color:#2a2c2b;margin:10px 0;">
                    <div style="font-size:11px;color:#e8e4dc;">{top_r['rec']}</div>
                    <div style="font-size:10px;color:#6b6e6c;">{f"{top_r['n_analysts']} analysts" if top_r['n_analysts'] else ""}{f" · {top_r['upside']:+.1f}%" if top_r['upside'] else ""}</div>
                </div>""",unsafe_allow_html=True)

    # ── DEEP DIVE ─────────────────────────────────────────
    with tab_deep:
        ci,cb=st.columns([3,1])
        with ci: dive_ticker=st.text_input("",placeholder="Enter ticker e.g. NVDA",label_visibility="collapsed").strip().upper()
        with cb: dive_run=st.button("Analyse →",key="dive_btn",use_container_width=True)
        if dive_run and dive_ticker:
            with st.spinner(f"Loading {dive_ticker}…"):
                info=get_info(dive_ticker); hist=get_hist(dive_ticker)
                sec=info.get("sector",""); th=get_thresh(sec)
                try:
                    score,signals=compute_score(hist,sec)
                except Exception:
                    score,signals=None,{}
            if not info.get("currentPrice") and not info.get("longName") and not info.get("shortName"):
                st.error(f"Could not find data for \"{dive_ticker}\". Check the ticker symbol and try again.")
                st.stop()
            name=info.get("longName",dive_ticker); price=info.get("currentPrice") or info.get("regularMarketPrice")
            beta=info.get("beta"); pe=info.get("trailingPE"); div=(info.get("dividendYield") or 0)*100
            mcap=info.get("marketCap"); target=info.get("targetMeanPrice")
            rec=analyst_label(info.get("recommendationKey","")); n_ana=info.get("numberOfAnalystOpinions")
            rec_mean=info.get("recommendationMean"); upside=((target-price)/price*100) if price and target else None
            desc=info.get("longBusinessSummary","")
            st.markdown(f"""<div style="margin-bottom:18px;">
                <div style="font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:#c8953a;margin-bottom:3px;">{sec}</div>
                <h2 style="font-family:'Fraunces',serif;font-size:1.7rem;font-weight:300;color:#e8e4dc;margin:0;">
                    {name} <span style="color:#6b6e6c;font-size:.9rem;">({dive_ticker})</span></h2>
                <div style="font-size:10px;color:#6b6e6c;margin-top:5px;">RSI: {th['rsi_os']}/{th['rsi_ob']} · CCI: {th['cci_os']}/{th['cci_ob']} · W%%R: {th['wpr_os']}/{th['wpr_ob']}</div>
            </div>""",unsafe_allow_html=True)
            for col,(lbl,val,sub) in zip(st.columns(6),[
                ("Price",f"${price:.2f}" if price else "—",""),("Market Cap",fmt_cap(mcap),""),
                ("Beta",f"{beta:.2f}" if beta else "—","vs market"),("P/E",f"{pe:.1f}" if pe else "—",""),
                ("Div Yield",f"{div:.2f}%",""),("Analyst Target",f"${target:.2f}" if target else "—",f"{upside:+.1f}%" if upside else "")]):
                with col: metric_card(lbl,val,sub)
            st.markdown("<br>",unsafe_allow_html=True)
            cg,cc=st.columns([1,2])
            with cg:
                if score: st.plotly_chart(score_gauge(score,dive_ticker),use_container_width=True,config={"displayModeBar":False})
                rc=("#4caf7d" if rec in("Strong Buy","Buy") else "#5bc8c8" if rec=="Hold" else "#e05c5c")
                st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;padding:14px;border-radius:3px;margin-top:4px;">
                    <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;margin-bottom:6px;">Analyst Sentiment</div>
                    <div style="font-family:'Fraunces',serif;font-size:1rem;color:{rc};">{rec}</div>
                    <div style="font-size:11px;color:#6b6e6c;margin-top:3px;">{f"{n_ana} analysts" if n_ana else ""}{f" · Mean: {rec_mean:.1f}/5" if rec_mean else ""}</div>
                </div>""",unsafe_allow_html=True)
            with cc:
                fp=price_chart(hist,dive_ticker)
                if fp: st.plotly_chart(fp,use_container_width=True,config={"displayModeBar":False})
            if signals:
                st.markdown("<br>",unsafe_allow_html=True)
                section_header("All 26 Indicators")
                st.plotly_chart(signals_chart(signals),use_container_width=True,config={"displayModeBar":False})
            if desc:
                st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;padding:18px;border-radius:3px;margin-top:6px;">
                    <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;margin-bottom:8px;">About</div>
                    <div style="font-size:12px;color:#a0a8a4;line-height:1.8;">{desc[:400]}{'…' if len(desc)>400 else ''}</div>
                </div>""",unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center;padding:60px 0;">
                <div style="font-size:3rem;margin-bottom:14px;">🔍</div>
                <div style="font-family:'Fraunces',serif;font-size:1.2rem;color:#6b6e6c;font-weight:300;">Enter any ticker for a full technical + fundamental analysis</div>
            </div>""",unsafe_allow_html=True)

    # ── COMPARE ───────────────────────────────────────────
    with tab_compare:
        c1,c2,c3=st.columns([2,2,1])
        with c1: t1=st.text_input("",placeholder="First ticker",label_visibility="collapsed",key="t1").strip().upper()
        with c2: t2=st.text_input("",placeholder="Second ticker",label_visibility="collapsed",key="t2").strip().upper()
        with c3: cmp_run=st.button("Compare →",key="cmp_btn",use_container_width=True)
        if cmp_run and t1 and t2:
            with st.spinner(f"Loading {t1} and {t2}…"):
                data={}
                for tk in [t1,t2]:
                    info=get_info(tk); hist=get_hist(tk); sec=info.get("sector","")
                    score,sigs=compute_score(hist,sec); data[tk]={"info":info,"hist":hist,"score":score,"signals":sigs,"sector":sec}
            st.markdown(f"""<div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:300;color:#e8e4dc;margin-bottom:20px;">
                {t1} <span style="color:#c8953a;">vs</span> {t2}</div>""",unsafe_allow_html=True)
            g1,g2=st.columns(2)
            with g1: st.plotly_chart(score_gauge(data[t1]["score"],t1),use_container_width=True,config={"displayModeBar":False})
            with g2: st.plotly_chart(score_gauge(data[t2]["score"],t2),use_container_width=True,config={"displayModeBar":False})
            s1,s2=data[t1]["score"] or 0,data[t2]["score"] or 0; winner=t1 if s1>s2 else t2
            st.markdown(f"""<div style="background:#181917;border:1px solid rgba(200,149,58,0.2);padding:12px 20px;
                border-radius:3px;margin-bottom:16px;text-align:center;">
                <span style="font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;">Technical Edge → </span>
                <span style="font-family:'Fraunces',serif;font-size:1rem;color:#c8953a;">{winner}</span>
                <span style="font-size:11px;color:#6b6e6c;"> by {abs(s1-s2):.1f} pts</span>
            </div>""",unsafe_allow_html=True)
            sec1,sec2=data[t1]["sector"],data[t2]["sector"]
            if sec1!=sec2:
                st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;padding:10px 16px;border-radius:3px;margin-bottom:12px;font-size:11px;color:#6b6e6c;">
                    ⚠ Different sectors — thresholds applied independently: <span style="color:#c8953a">{t1} ({sec1})</span> / <span style="color:#c8953a">{t2} ({sec2})</span>
                </div>""",unsafe_allow_html=True)
            section_header("Fundamentals")
            def gv(tk,key): return data[tk]["info"].get(key)
            rows_c=[{"Metric":lbl,t1:fn(t1),t2:fn(t2)} for lbl,fn in [
                ("Price",     lambda tk:f"${gv(tk,'currentPrice'):.2f}" if gv(tk,'currentPrice') else "—"),
                ("Market Cap",lambda tk:fmt_cap(gv(tk,'marketCap'))),
                ("Beta",      lambda tk:f"{gv(tk,'beta'):.2f}" if gv(tk,'beta') else "—"),
                ("P/E",       lambda tk:f"{gv(tk,'trailingPE'):.1f}" if gv(tk,'trailingPE') else "—"),
                ("Div Yield", lambda tk:f"{(gv(tk,'dividendYield') or 0)*100:.2f}%"),
                ("Target",    lambda tk:f"${gv(tk,'targetMeanPrice'):.2f}" if gv(tk,'targetMeanPrice') else "—"),
                ("Consensus", lambda tk:analyst_label(gv(tk,'recommendationKey') or "")),
                ("Sector",    lambda tk:gv(tk,'sector') or "—"),]]
            st.dataframe(pd.DataFrame(rows_c).set_index("Metric").style.set_properties(**{
                "background-color":"#111210","color":"#e8e4dc","border-color":"#1f2020",
                "font-family":"DM Mono, monospace","font-size":"12px"}).set_table_styles([
                {"selector":"th","props":[("background-color","#0b0a09"),("color","#6b6e6c"),("font-size","9px"),
                 ("letter-spacing","0.15em"),("text-transform","uppercase"),("border-color","#1f2020"),("padding","10px 14px")]},
                {"selector":"td","props":[("padding","10px 14px"),("border-color","#1f2020")]}]),use_container_width=True)
            st.markdown("<br>",unsafe_allow_html=True)
            a1,a2=st.columns(2)
            for col,tk in [(a1,t1),(a2,t2)]:
                with col:
                    rec=analyst_label(data[tk]["info"].get("recommendationKey",""))
                    rc=("#4caf7d" if rec in("Strong Buy","Buy") else "#5bc8c8" if rec=="Hold" else "#e05c5c")
                    n=data[tk]["info"].get("numberOfAnalystOpinions")
                    tgt=data[tk]["info"].get("targetMeanPrice"); prc=data[tk]["info"].get("currentPrice")
                    up=f"{((tgt-prc)/prc*100):+.1f}%" if tgt and prc else "—"
                    st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;padding:14px;border-radius:3px;">
                        <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;margin-bottom:5px;">{tk} · Consensus</div>
                        <div style="font-family:'Fraunces',serif;color:{rc};">{rec}</div>
                        <div style="font-size:11px;color:#6b6e6c;margin-top:3px;">{f"{n} analysts · " if n else ""}Target: {up}</div>
                    </div>""",unsafe_allow_html=True)
            sig1,sig2=data[t1]["signals"],data[t2]["signals"]
            if sig1 and sig2:
                disagree={k for k in sig1 if sig1.get(k,0)!=sig2.get(k,0)}
                if disagree:
                    st.markdown("<br>",unsafe_allow_html=True); section_header("Where They Disagree")
                    d1,d2=st.columns(2)
                    for col,tk,smap in [(d1,t1,sig1),(d2,t2,sig2)]:
                        with col:
                            bull=[k for k in disagree if smap.get(k,0)==1]
                            bear=[k for k in disagree if smap.get(k,0)==-1]
                            st.markdown(f"""<div style="background:#181917;border:1px solid #2a2c2b;padding:14px;border-radius:3px;">
                                <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;margin-bottom:8px;">{tk}</div>
                                {f'<div style="font-size:10px;color:#4caf7d;margin-bottom:3px;">▲ Bullish on</div><div style="font-size:11px;color:#a0a8a4;margin-bottom:8px;">{", ".join(bull)}</div>' if bull else ""}
                                {f'<div style="font-size:10px;color:#e05c5c;margin-bottom:3px;">▼ Bearish on</div><div style="font-size:11px;color:#a0a8a4;">{", ".join(bear)}</div>' if bear else ""}
                            </div>""",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True); section_header("Normalised 1-Year Price Performance")
            fig_c=go.Figure()
            for tk,color in [(t1,"#c8953a"),(t2,"#5bc8c8")]:
                hh=data[tk]["hist"]
                if hh is not None:
                    norm=hh["Close"]/hh["Close"].iloc[0]*100
                    fig_c.add_trace(go.Scatter(x=hh.index,y=norm,name=tk,line=dict(color=color,width=2)))
            layout=CHART_LAYOUT.copy(); layout.update(height=250,margin=dict(l=0,r=0,t=10,b=0))
            fig_c.update_layout(**layout)
            st.plotly_chart(fig_c,use_container_width=True,config={"displayModeBar":False})
        else:
            st.markdown("""<div style="text-align:center;padding:60px 0;">
                <div style="font-size:3rem;margin-bottom:14px;">⚔</div>
                <div style="font-family:'Fraunces',serif;font-size:1.2rem;color:#6b6e6c;font-weight:300;">Enter two ticker symbols to compare head-to-head</div>
            </div>""",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  FINANCIAL MODELING PAGE
# ─────────────────────────────────────────────────────────
def render_models():
    st.markdown("""<div style="padding:24px 0 18px;border-bottom:1px solid #1f2020;margin-bottom:24px;">
        <div style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#c8953a;margin-bottom:5px;">Torosian Stock Insights</div>
        <h1 style="font-family:'Fraunces',serif;font-size:1.9rem;font-weight:300;color:#e8e4dc;margin:0;">
            Financial <em style="color:#c8953a;">Modeling</em></h1>
        <p style="color:#6b6e6c;font-size:11px;margin-top:6px;">Professional-grade models — enter any ticker to run the analysis</p>
    </div>""",unsafe_allow_html=True)

    tab_trend,tab_psych,tab_value = st.tabs([
        "📈  The Trend Models",
        "🧠  The Psychological Models",
        "💎  The Value Models",
    ])

    with tab_trend:
        st.markdown("<br>",unsafe_allow_html=True)
        model_tab1,model_tab2,model_tab3 = st.tabs([
            "Stage Analysis","Mansfield Relative Strength","Mean Reversion"])
        with model_tab1:
            try: model_stage_analysis()
            except Exception as e: st.error(f"Stage Analysis error: {type(e).__name__}: {e}")
        with model_tab2:
            try: model_mansfield_rs()
            except Exception as e: st.error(f"Mansfield RS error: {type(e).__name__}: {e}")
        with model_tab3:
            try: model_mean_reversion()
            except Exception as e: st.error(f"Mean Reversion error: {type(e).__name__}: {e}")

    with tab_psych:
        st.markdown("<br>",unsafe_allow_html=True)
        model_tab4,model_tab5 = st.tabs(["Elliott Wave + Fibonacci","Volume Profile + VWAP"])
        with model_tab4:
            try: model_elliott_wave()
            except Exception as e: st.error(f"Elliott Wave error: {type(e).__name__}: {e}")
        with model_tab5:
            try: model_volume_profile()
            except Exception as e: st.error(f"Volume Profile error: {type(e).__name__}: {e}")

    with tab_value:
        st.markdown("<br>",unsafe_allow_html=True)
        model_tab6,model_tab7 = st.tabs(["Discounted Cash Flow","CAN SLIM"])
        with model_tab6:
            try: model_dcf()
            except Exception as e: st.error(f"DCF error: {type(e).__name__}: {e}")
        with model_tab7:
            try: model_canslim()
            except Exception as e: st.error(f"CAN SLIM error: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────
#  GLOBAL SIDEBAR  (always renders regardless of page)
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 16px;">
        <div style="font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:#c8953a;margin-bottom:5px;">Equity Research Platform</div>
        <div style="font-family:'Fraunces',serif;font-size:1.3rem;font-weight:300;color:#e8e4dc;line-height:1.2;">
            Torosian<br><em style="color:#c8953a;">Stock Insights</em></div>
    </div>
    <hr style="border-color:#1f2020;margin:0 0 14px;">
    <div style="font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#6b6e6c;margin-bottom:10px;">Navigation</div>
    """,unsafe_allow_html=True)

    pages = [("📖  Learn", "learn"), ("📊  Screener & Analysis", "app"), ("🧮  Financial Modeling", "models")]
    for label, key in pages:
        active = st.session_state.page == key
        btn_style = "background:#c8953a!important;color:#0b0a09!important;" if active else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("<hr style='border-color:#1f2020;margin:14px 0;'>",unsafe_allow_html=True)

    # Screener filters only show on screener page — but sidebar always renders
    if st.session_state.page == "app":
        pass  # filters rendered inside render_app() via render_screener_sidebar()
    elif st.session_state.page == "learn":
        st.markdown("""<div style="font-size:10px;color:#6b6e6c;line-height:1.8;padding-top:4px;">
            26 technical indicators<br>Sector-adjusted thresholds<br>7 financial models<br>
            Live data via Yahoo Finance</div>""",unsafe_allow_html=True)
    elif st.session_state.page == "models":
        st.markdown("""<div style="font-size:10px;color:#6b6e6c;line-height:1.8;padding-top:4px;">
            <b style="color:#a0a8a4;">Trend Models</b><br>
            Stage Analysis<br>Mansfield RS<br>Mean Reversion<br><br>
            <b style="color:#a0a8a4;">Psychological Models</b><br>
            Elliott Wave + Fibonacci<br>Volume Profile + VWAP<br><br>
            <b style="color:#a0a8a4;">Value Models</b><br>
            Discounted Cash Flow<br>CAN SLIM
        </div>""",unsafe_allow_html=True)

    st.markdown("""<hr style="border-color:#1f2020;margin:14px 0 10px;">
    <div style="font-size:9px;color:#3a3c3b;line-height:1.8;">Data via Yahoo Finance<br>Not financial advice</div>""",
    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────
page = st.session_state.page
if page == "learn":
    render_learn()
elif page == "app":
    render_app()
elif page == "models":
    render_models()
