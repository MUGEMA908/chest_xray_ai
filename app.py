"""
app.py - Streamlit interface for the Chest X-ray AI Assistant.
Run:  streamlit run app.py
Languages: English / Kinyarwanda (text lives in i18n.py).  Themes: light / dark.
"""
import hashlib, io, json, os
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
from engine import XrayAI, PNEU_SIGNS, image_quality_check, risk_level
from i18n import TEXT

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Chest X-ray AI Assistant", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------ language + theme switches (top of sidebar)
with st.sidebar:
    brand_text = TEXT.get(st.session_state.get("lang", "en"), TEXT["en"])
    st.markdown(
        f'<div class="sidebar-brand"><div class="sidebar-brand-mark">🫁</div>'
        f'<div><div class="sidebar-brand-title">{brand_text["sidebar_brand"]}</div>'
        f'<div class="sidebar-brand-sub">{brand_text["sidebar_local"]}</div></div></div>',
        unsafe_allow_html=True,
    )
    lang = st.radio("Language / Ururimi", ["en", "rw"], format_func=lambda c: TEXT[c]["lang_name"], horizontal=True, key="lang")
T = TEXT[lang]


# Streamlit treats a widget whose label/options change (language switch) as a NEW widget and resets it.
# These helpers remember each choice ourselves, so settings survive a language change.
def _cur(name, options):
    v = st.session_state.get("p_" + name, options[0])
    return v if v in options else options[0]


def pradio(name, label, options, fmt, **kw):
    v = st.radio(label, options, index=options.index(_cur(name, options)), format_func=fmt, key=f"{name}_{lang}", **kw)
    st.session_state["p_" + name] = v
    return v


def pselect(name, label, options, fmt, **kw):
    v = st.selectbox(label, options, index=options.index(_cur(name, options)), format_func=fmt, key=f"{name}_{lang}", **kw)
    st.session_state["p_" + name] = v
    return v


def pslider(name, label, lo, hi, default, step, **kw):
    v = st.slider(label, lo, hi, st.session_state.get("p_" + name, default), step, key=f"{name}_{lang}", **kw)
    st.session_state["p_" + name] = v
    return v

with st.sidebar:
    theme = pradio("theme", T["theme"], ["light", "dark"], lambda v: T["theme_" + v], horizontal=True)

PALETTE = {
    "light": dict(bg="#F4F7FB", panel="#FFFFFF", side="#EDF3F6", ink="#14212E", mute="#5F7280", line="#D8E1E8", track="#EAF0F4",
                  teal="#138B84", red="#D64E4E", amber="#B7781B", film="#0E1B23"),
    "dark":  dict(bg="#09131A", panel="#111E29", side="#0D171F", ink="#EAF3F9", mute="#9EB1BD", line="#243744", track="#1E2D39",
                  teal="#4AD6C9", red="#F17073", amber="#E5B15C", film="#050D12"),
}[theme]
root = ":root{" + "".join(f"--{k}:{v};" for k, v in PALETTE.items()) + f"color-scheme:{theme};}}"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600;700&display=swap');
html, body, .stApp { font-family:'IBM Plex Sans', sans-serif; }
html { scroll-behavior:smooth; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background:radial-gradient(circle at top left, rgba(19,139,132,0.12), transparent 30%),
               radial-gradient(circle at top right, rgba(214,78,78,0.09), transparent 26%),
               var(--bg);
}
[data-testid="stHeader"] { background:transparent; }
[data-testid="stHeader"] > div {
    justify-content:space-between; align-items:center; gap:.75rem; padding:0.25rem 0 0.35rem;
}
[data-testid="stHeader"] button,
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
button[kind="secondary"],
button[kind="primary"] {
    border-radius:10px !important; border:1px solid var(--line) !important;
    background:linear-gradient(180deg, var(--panel), color-mix(in srgb, var(--panel) 88%, var(--track))) !important;
    color:var(--ink) !important; box-shadow:0 4px 10px rgba(8,16,21,0.04) !important; font-weight:600 !important; }
[data-testid="stHeader"] button:hover,
[data-testid="stSidebarCollapseButton"]:hover,
button[kind="header"]:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover {
    border-color:color-mix(in srgb, var(--teal) 45%, var(--line)) !important;
    box-shadow:0 6px 14px rgba(19,139,132,0.08) !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background:linear-gradient(180deg, var(--side), color-mix(in srgb, var(--side) 88%, var(--panel)));
    border-right:1px solid var(--line); box-shadow:inset -1px 0 0 rgba(255,255,255,0.04);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.65rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] li { color:var(--ink); }
.sidebar-brand {
    display:flex; align-items:center; gap:.75rem; padding:.8rem .85rem; margin:.15rem 0 .25rem;
    border:1px solid var(--line); border-radius:14px;
    background:linear-gradient(135deg, color-mix(in srgb, var(--teal) 12%, var(--panel)), var(--panel));
    box-shadow:0 8px 20px rgba(9,19,26,0.05);
}
.sidebar-brand-mark {
    display:flex; align-items:center; justify-content:center; width:42px; height:42px; flex:0 0 42px;
    border-radius:12px; background:color-mix(in srgb, var(--teal) 16%, var(--panel)); font-size:1.45rem;
}
.sidebar-brand-title { color:var(--ink); font-size:.92rem; font-weight:700; line-height:1.2; }
.sidebar-brand-sub { color:var(--mute); font-size:.72rem; line-height:1.3; margin-top:.2rem; }
.st-key-sidebar-source, .st-key-sidebar-display {
    border:1px solid var(--line); border-radius:14px; padding:.75rem .8rem;
    background:var(--panel);
    box-shadow:0 6px 16px rgba(9,19,26,0.035);
}
.st-key-sidebar-source h3, .st-key-sidebar-display h3 {
    margin:.05rem 0 .35rem; font-size:1rem; letter-spacing:-.01em;
}
.st-key-sidebar-source [data-testid="stRadio"] [role="radiogroup"],
.st-key-sidebar-display [data-testid="stRadio"] [role="radiogroup"] { gap:.35rem; }
.st-key-sidebar-source [data-testid="stRadioOption"],
.st-key-sidebar-display [data-testid="stRadioOption"] {
    padding:.42rem .55rem; border:1px solid transparent; border-radius:9px;
    transition:background .16s ease, border-color .16s ease;
}
.st-key-sidebar-source [data-testid="stRadioOption"]:hover,
.st-key-sidebar-display [data-testid="stRadioOption"]:hover {
    border-color:var(--line); background:color-mix(in srgb, var(--teal) 7%, var(--panel));
}
.st-key-sidebar-source [data-testid="stRadioOption"][data-selected="true"],
.st-key-sidebar-display [data-testid="stRadioOption"][data-selected="true"] {
    border-color:color-mix(in srgb, var(--teal) 34%, var(--line));
    background:color-mix(in srgb, var(--teal) 9%, var(--panel));
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:var(--ink); }
h1, h2, h3 { font-family:'IBM Plex Serif', serif !important; letter-spacing:-0.02em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--mute); }
.block-container { padding-top:1.5rem; max-width:1360px; }
#MainMenu, footer { visibility:hidden; }

/* premium hero */
.hero {
    display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
    background:linear-gradient(135deg, color-mix(in srgb, var(--panel) 94%, transparent), color-mix(in srgb, var(--track) 84%, var(--panel)));
    border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem; margin-bottom:1.15rem;
    box-shadow:0 12px 28px rgba(9,19,26,0.06);
}
.hero h1 { margin:0; font-size:2.2rem; line-height:1.08; color:var(--ink); }
.hero p { margin:.4rem 0 0; color:var(--mute); font-size:.98rem; max-width:700px; }
.eyebrow {
    display:inline-block; padding:.22rem .6rem; border-radius:999px; font-size:.72rem; letter-spacing:.08em;
    text-transform:uppercase; font-weight:700; color:var(--teal); background:color-mix(in srgb, var(--teal) 10%, var(--panel));
    border:1px solid color-mix(in srgb, var(--teal) 30%, var(--panel)); margin-bottom:.55rem;
}
.stamp {
    border:1px solid color-mix(in srgb, var(--red) 42%, var(--panel)); background:color-mix(in srgb, var(--red) 10%, var(--panel));
    color:var(--red); padding:.5rem .8rem; border-radius:12px; font-size:.82rem; font-weight:700;
    max-width:300px; line-height:1.3; box-shadow:inset 0 0 0 1px rgba(255,255,255,0.04);
}

/* containers */
.st-key-viewer { background:linear-gradient(180deg, color-mix(in srgb, var(--film) 95%, var(--panel)), var(--film)); border:1px solid rgba(255,255,255,0.04); border-radius:14px; padding:.8rem; box-shadow:0 12px 24px rgba(0,0,0,0.12); }
.st-key-viewer p { color:#9FB1BD !important; font-size:.82rem; }
.st-key-findings { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; }
.panel { background:linear-gradient(180deg, var(--panel), color-mix(in srgb, var(--panel) 92%, var(--track))); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; color:var(--ink); box-shadow:0 10px 22px rgba(10,18,24,0.04); }
.note { color:var(--mute); font-size:.85rem; }

/* verdict card */
.verdict {
    border:1px solid color-mix(in srgb, var(--c) 26%, var(--line)); border-left:6px solid var(--c);
    background:linear-gradient(180deg, color-mix(in srgb, var(--c) 14%, var(--panel)), color-mix(in srgb, var(--c) 6%, var(--panel)));
    border-radius:14px; padding:1rem 1.2rem; color:var(--ink); box-shadow:0 12px 25px rgba(13,23,31,0.05);
}
.verdict.pneu { --c:var(--red); } .verdict.norm { --c:var(--teal); } .verdict.unc { --c:var(--amber); }
.verdict .big { font-family:'IBM Plex Serif', serif; font-size:1.9rem; font-weight:700; color:var(--c); margin:.25rem 0; line-height:1.15; }

/* bars */
.track { position:relative; height:9px; background:var(--track); border-radius:5px; margin-top:.35rem; overflow:hidden; }
.fill { height:9px; border-radius:5px; box-shadow:0 0 10px rgba(255,255,255,0.12); }
.mark { position:absolute; top:-3px; width:2px; height:15px; background:var(--ink); opacity:.65; }
.row { margin:.7rem 0 .8rem; } .row .top { display:flex; justify-content:space-between; gap:.6rem; font-size:.96rem; color:var(--ink); }
.row .name { font-weight:700; } .row .desc { color:var(--mute); font-size:.8rem; margin-top:.15rem; }
.chip { display:inline-block; padding:.22rem .7rem; margin:.18rem .28rem .15rem 0; border-radius:999px; font-size:.82rem; font-weight:600;
        color:var(--red); background:color-mix(in srgb, var(--red) 12%, var(--panel)); border:1px solid color-mix(in srgb, var(--red) 32%, var(--panel)); }
.chip.ok { color:var(--teal); background:color-mix(in srgb, var(--teal) 12%, var(--panel)); border-color:color-mix(in srgb, var(--teal) 34%, var(--panel)); }

/* tips */
.tips { margin:.2rem 0 0; padding-left:1.1rem; } .tips li { margin:.5rem 0; color:var(--ink); line-height:1.5; }
.tips li.urgent { list-style:none; margin-left:-1.1rem; padding:.65rem .8rem; border-radius:10px; font-weight:600;
        background:color-mix(in srgb, var(--red) 13%, var(--panel)); border-left:5px solid var(--red); }
.sys { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:.95rem 1rem; color:var(--ink); font-family:'IBM Plex Mono', monospace; font-size:.85rem; white-space:pre-wrap; }

/* streamlit widgets */
[data-baseweb="select"] > div, [data-baseweb="input"] > div { background:var(--panel); border-color:var(--line); color:var(--ink); border-radius:10px; }
[data-baseweb="select"] *, [data-baseweb="input"] input { color:var(--ink); }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background:var(--panel); color:var(--ink); }
[data-baseweb="popover"] li:hover { background:var(--track); }
[data-testid="stFileUploaderDropzone"] { background:linear-gradient(180deg, var(--panel), var(--track)); border:1px dashed var(--line); border-radius:12px; }
[data-testid="stFileUploaderDropzone"] * { color:var(--ink); }
[data-testid="stFileUploaderDropzone"] button, .stDownloadButton button { background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:10px; }
.stDownloadButton button:hover { border-color:var(--teal); color:var(--teal); }
[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:var(--ink); }
[data-testid="stMetric"] * { color:var(--ink); } [data-testid="stMetricValue"] { font-family:'IBM Plex Serif', serif; }
[data-testid="stAlert"] { background:color-mix(in srgb, var(--amber) 14%, var(--panel)); border-radius:10px; }
[data-testid="stAlert"] * { color:var(--ink); }
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {
    display:flex; gap:.5rem; padding:.35rem; margin:.15rem 0 .4rem;
    border:1px solid var(--line); border-radius:14px;
    background:color-mix(in srgb, var(--panel) 88%, var(--track));
    box-shadow:0 8px 20px rgba(10,17,23,0.05);
}
.stTabs [data-baseweb="tab"], .stTabs [role="tab"] {
    flex:1 1 0; min-height:48px; justify-content:center; padding:.55rem .75rem;
    border:1px solid transparent; border-radius:10px; color:var(--mute);
    font-size:.94rem; font-weight:700; line-height:1.25; transition:all .18s ease;
}
.stTabs [data-baseweb="tab"]:hover, .stTabs [role="tab"]:hover {
    color:var(--ink); background:color-mix(in srgb, var(--teal) 7%, var(--panel));
}
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [role="tab"][aria-selected="true"] {
    color:var(--panel); border-color:color-mix(in srgb, var(--teal) 80%, var(--ink));
    background:linear-gradient(135deg, var(--teal), color-mix(in srgb, var(--teal) 78%, var(--ink)));
    box-shadow:0 5px 12px color-mix(in srgb, var(--teal) 22%, transparent);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] *, .stTabs [role="tab"][aria-selected="true"] * { color:var(--panel) !important; }
.stTabs [data-baseweb="tab"] > div, .stTabs [role="tab"] > div {
    display:flex; align-items:center; justify-content:center; gap:.45rem;
}
[data-testid="stTable"] table { color:var(--ink); background:var(--panel); border-radius:12px; overflow:hidden; } [data-testid="stTable"] th { background:var(--track); color:var(--ink); }
[data-testid="stTable"] td, [data-testid="stTable"] th { border-color:var(--line); }
[data-testid="stRadioOption"]:not([data-selected="true"]) > div > div:first-child { background:var(--panel) !important; border-color:var(--mute) !important; }
[data-testid="stSelectbox"] [role="group"] { background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:10px; }
[data-testid="stSelectbox"] [role="group"] *, [data-testid="stSelectbox"] input { background-color:transparent !important; color:var(--ink) !important; }
[data-testid="stSelectbox"] svg { fill:var(--mute); color:var(--mute); }
[role="listbox"] { background:var(--panel) !important; border:1px solid var(--line); border-radius:10px; }
[role="listbox"] [role="option"], [role="listbox"] [role="option"] * { background:transparent !important; color:var(--ink) !important; }
[role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][data-focused="true"], [role="listbox"] [role="option"][aria-selected="true"] { background:var(--track) !important; }
[data-testid="stSlider"] * { color:var(--ink); }
[data-testid="stAlert"] { border-radius:8px; }
img { border-radius:12px; }
div[data-testid="stImage"] { overflow:hidden; border-radius:14px; }
"""
st.markdown(f"<style>{root}{CSS}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_ai():
    return XrayAI()


with st.spinner(T["loading"]):
    ai = load_ai()


def results_dir():
    for d in ("results_kaggle", "results"):
        p = os.path.join(HERE, d, "thresholds.json")
        if os.path.exists(p):
            return d, json.load(open(p))
    return None, {"lo": 0.46, "hi": 0.66}


RES_DIR, TH = results_dir()
LO, HI = TH["lo"], TH["hi"]
COLOR = {"High": "var(--red)", "Moderate": "var(--amber)", "Low": "var(--mute)", "Minimal": "var(--teal)"}
fname = lambda k: k.replace("_", " ")

# ------------------------------------------------------------ header
st.markdown(
    f'<div class="hero"><div><div class="eyebrow">AI diagnostics</div><h1>{T["app_title"]}</h1><p>{T["app_sub"]}</p></div><div class="stamp">{T["stamp"]}</div></div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ sidebar
img, image_bytes, ref_label, name = None, None, None, None
with st.sidebar:
    with st.container(key="sidebar-source"):
        st.subheader(T["src_title"])
        src = pradio("src", "src", ["upload", "sample"], lambda v: T["src_" + v], label_visibility="collapsed")
        if src == "upload":
            st.markdown(f'<p class="note" style="margin:0">{T["upload_label"]}</p>', unsafe_allow_html=True)
            up = st.file_uploader("upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="upload")
            if up:
                image_bytes, name = up.getvalue(), up.name
                img = Image.open(io.BytesIO(image_bytes))
        else:
            lp = os.path.join(HERE, "sample_images", "labels.csv")
            if os.path.exists(lp):
                lab = pd.read_csv(lp).set_index("file")
                pick = pselect("sample", T["sample_label"], list(lab.index),
                               lambda f: f"{T['sample_' + lab.loc[f, 'kind'].lower()]} {lab.loc[f, 'n']}")
                with open(os.path.join(HERE, "sample_images", pick), "rb") as sample_file:
                    image_bytes = sample_file.read()
                img, name = Image.open(io.BytesIO(image_bytes)), pick
                ref_label = T["ref_" + lab.loc[pick, "dataset_label"]]
            else:
                st.info(T["no_samples"])
    with st.sidebar:
        with st.container(key="sidebar-display"):
            st.subheader(T["display"])
            mode = pradio("mode", "mode", ["simple", "detail"], lambda v: T["mode_" + v], label_visibility="collapsed")
            simple = mode == "simple"
            thr, n_show = 0.5, 8
            if not simple:
                st.subheader(T["settings"])
                thr = pslider("thr", T["thr_label"], 0.20, 0.90, 0.50, 0.05, help=T["thr_help"])
                n_show = pslider("nshow", T["n_show"], 3, 18, 8, 1)
    st.caption(T["model_caption"])

tab_a, tab_b, tab_c = st.tabs([T["tab_analyse"], T["tab_perf"], T["tab_limits"]])

# ------------------------------------------------------------ analyse
with tab_a:
    if img is None:
        st.markdown(f'<div class="panel">{T["start_hint"]}</div>', unsafe_allow_html=True)
    else:
        image_key = hashlib.blake2b(image_bytes, digest_size=16).hexdigest()
        cached_quality = st.session_state.get("_xray_quality")
        if cached_quality is None or cached_quality["key"] != image_key:
            quality_warnings = image_quality_check(img)
            st.session_state["_xray_quality"] = {"key": image_key, "warnings": quality_warnings}
        else:
            quality_warnings = cached_quality["warnings"]
        for key, arg in quality_warnings:
            st.warning(T[key].format(r=arg))
        cached_result = st.session_state.get("_xray_prediction")
        if cached_result is None or cached_result["key"] != image_key:
            res = ai.predict(img)
            st.session_state["_xray_prediction"] = {"key": image_key, "result": res}
            st.session_state.pop("_xray_gradcam", None)
        else:
            res = cached_result["result"]
        ranked = sorted(res.probs.items(), key=lambda kv: -kv[1])
        flagged = [(k, v) for k, v in ranked if v >= thr]
        sign_score = max(res.probs[k] for k in PNEU_SIGNS)
        verdict = "pneu" if sign_score >= HI else ("norm" if sign_score <= LO else "unc")

        left, right = st.columns([1.05, 1], gap="large")
        with left:
            if simple:
                focus = max(PNEU_SIGNS, key=lambda k: res.probs[k])
            else:
                focus = st.selectbox(T["focus_label"], [k for k, _ in ranked], format_func=fname, index=0, help=T["focus_help"])
            view = pradio("view", "view", ["orig", "heat"], lambda v: T["view_" + v], horizontal=True, label_visibility="collapsed")
            if view == "heat":
                cached_cam = st.session_state.get("_xray_gradcam")
                if cached_cam is None or cached_cam["key"] != image_key or cached_cam["focus"] != focus:
                    cam = ai.gradcam(res.tensor, focus)
                    st.session_state["_xray_gradcam"] = {"key": image_key, "focus": focus, "cam": cam}
                else:
                    cam = cached_cam["cam"]
                shown, cap = ai.overlay(res.image224, cam), T["cap_heat"].format(f=fname(focus))
            else:
                shown, cap = res.image224, T["cap_orig"]
            with st.container(key="viewer"):
                st.image(shown, width="stretch", clamp=True)
                st.markdown(f"{cap} · {res.seconds * 1000:.0f} ms")
            if ref_label:
                st.markdown(f'<p class="note">{T["ref_label"]} <b>{ref_label}</b></p>', unsafe_allow_html=True)

        with right:
            if simple:
                st.markdown(
                    f'<div class="verdict {verdict}"><div class="note">{T["result_label"]}</div>'
                    f'<div class="big">{T["v_" + verdict]}</div><div>{T["v_" + verdict + "_msg"]}</div>'
                    f'<div class="track" style="margin-top:.9rem"><div class="fill" style="width:{sign_score*100:.0f}%;background:var(--c)"></div>'
                    f'<div class="mark" style="left:{LO*100:.0f}%"></div><div class="mark" style="left:{HI*100:.0f}%"></div></div>'
                    f'<div class="note" style="margin-top:.4rem">{T["score_line"].format(s=f"{sign_score:.0%}")}</div></div>',
                    unsafe_allow_html=True)
                st.markdown(f"**{T['how_title']}**")
                st.markdown(T["how_steps"])

        # ---- detailed findings (inside the right column in detail mode, in an expander in simple mode)
        holder = st.expander(T["expander_all"]) if simple else right
        with holder:
            with st.container(key="findings"):
                st.markdown(f"### {T['findings_title']}")
                if flagged:
                    st.markdown("".join(f'<span class="chip">{fname(k)} {v:.0%}</span>' for k, v in flagged[:8]), unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="chip ok">{T["none_flagged"]}</span><p class="note">{T["none_note"]}</p>', unsafe_allow_html=True)
                html = ""
                for k, v in ranked[:(18 if simple else n_show)]:
                    r = risk_level(v)
                    html += (f'<div class="row"><div class="top"><span class="name">{fname(k)}</span>'
                             f'<span style="color:{COLOR[r]};font-weight:600">{v:.0%} · {T["s_" + r]}</span></div>'
                             f'<div class="track"><div class="fill" style="width:{v*100:.0f}%;background:{COLOR[r]}"></div>'
                             f'<div class="mark" style="left:{thr*100:.0f}%"></div></div>'
                             f'<div class="desc">{T["desc"].get(k, "")}</div></div>')
                st.markdown(html, unsafe_allow_html=True)
                st.markdown(f'<p class="note">{T["thr_note"].format(t=f"{thr:.2f}")}</p>', unsafe_allow_html=True)
                if lang == "rw":
                    st.markdown(f'<p class="note">{T["findings_names_note"]}</p>', unsafe_allow_html=True)

        # ---- patient tips (simple mode)
        tips = T["tips_" + {"pneu": "pneu", "norm": "norm", "unc": "unc"}[verdict]]
        if simple:
            li = "".join(f'<li class="urgent">{t[1:]}</li>' if t.startswith("!") else f"<li>{t}</li>" for t in tips)
            st.markdown(f'<div class="panel" style="margin-top:1rem"><h3 style="margin:0 0 .2rem">{T["tips_title"]}</h3>'
                        f'<p class="note" style="margin:0 0 .4rem">{T["tips_note"]}</p><ul class="tips">{li}</ul></div>', unsafe_allow_html=True)

        # ---- downloads
        df = pd.DataFrame({"finding": [k for k, _ in ranked], "score": [round(v, 4) for _, v in ranked], "flagged": [v >= 0.5 for _, v in ranked]})
        buf = io.StringIO()
        buf.write(f"{T['report_title']}\n{T['report_image']}: {name}\nModel: DenseNet121 (torchxrayvision, densenet121-res224-all)\n\n")
        buf.write(f"{T['report_result']}: {T['v_' + verdict]} ({sign_score:.0%})\n")
        buf.write(f"{T['report_flagged']}: {', '.join(fname(k) for k, v in ranked if v >= 0.5) or T['report_none']}\n\n{df.to_string(index=False)}\n\n")
        buf.write(f"{T['report_tips']}:\n" + "\n".join(f"- {t.lstrip('!')}" for t in tips) + f"\n\n{T['report_foot']}\n")
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button(T["dl_report"], buf.getvalue(), "xray_report.txt", use_container_width=True)
        c2.download_button(T["dl_scores"], df.to_csv(index=False), "xray_scores.csv", use_container_width=True)

# ------------------------------------------------------------ performance
with tab_b:
    st.markdown(f"### {T['perf_title']}")
    st.markdown(T["perf_intro"])
    mp = os.path.join(HERE, RES_DIR or "results", "metrics.csv")
    if RES_DIR and os.path.exists(mp):
        m = pd.read_csv(mp)
        st.markdown(f'<p class="note">{T["perf_source"].format(src=TH.get("source", ""), nn=TH.get("n_normal", "?"), np=TH.get("n_pneumonia", "?"))}</p>', unsafe_allow_html=True)
        best = m[m.score.str.startswith("Combined")].iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric("AUROC", f"{best['AUROC']:.2f}"); b.metric(T["m_sens"], f"{best['recall']:.0%}")
        c.metric(T["m_spec"], f"{best['specificity']:.0%}"); d.metric("F1", f"{best['F1']:.2f}")
        st.caption(T["perf_caption"])
        st.table(m.assign(row=m.score + " @ " + m.threshold.astype(str)).set_index("row").drop(columns=["score", "threshold"]).rename_axis(None))
        vp = os.path.join(HERE, RES_DIR, "verdict_counts.csv")
        if os.path.exists(vp):
            vc = pd.read_csv(vp); st.markdown(f"**{T['perf_verdicts']}**"); st.table(vc.set_index(vc.columns[0]).rename_axis(None))
        i1, i2 = st.columns(2)
        for col, f in [(i1, "confusion_matrix.png"), (i2, "roc_curve.png")]:
            p = os.path.join(HERE, RES_DIR, f)
            if os.path.exists(p): col.image(p, width="stretch")
        st.markdown(f'<p class="note">{T["perf_caveat"]}</p>', unsafe_allow_html=True)
    else:
        st.info(T["perf_none"])

# ------------------------------------------------------------ limits
with tab_c:
    st.markdown(f"### {T['limits_title']}")
    st.markdown(T["limits_body"])
    st.markdown(f"**{T['limits_head']}**")
    st.markdown("\n".join(f"- {x}" for x in T["limits_list"]))
    st.markdown(T["limits_reduce"])
    st.markdown(f"### {T['design_title']}")
    st.markdown(f'<div class="sys">{T["design"]}</div>', unsafe_allow_html=True)
