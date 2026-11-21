from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="MolInsight | OraLabs Predict",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
      color-scheme: dark;
      --mi-bg: #0c1016;
      --mi-surface: #141a22;
      --mi-surface-raised: #1a222d;
      --mi-border: #293341;
      --mi-text: #edf2f7;
      --mi-muted: #9aa8b8;
      --mi-accent: #5a9de0;
      --mi-accent-soft: #1c3550;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
      color-scheme: dark;
      max-width: 100%;
      min-width: 0;
      overflow-x: clip;
      background: var(--mi-bg);
    }
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
      min-width: 0;
      max-width: 100%;
    }
    [data-testid="stSidebar"], button[data-testid="stBaseButton-headerNoPadding"] {
      display: none !important;
    }
    [data-testid="stDeployButton"], [data-testid="stAppDeployButton"], .stAppDeployButton {
      display: none !important;
      visibility: hidden !important;
    }
    .stMainBlockContainer, .block-container {
      box-sizing: border-box;
      width: 100%;
      max-width: 1160px;
      margin-inline: auto;
      padding: .75rem clamp(1rem, 3vw, 2rem) 3rem !important;
    }
    h1, h2, h3, h4 { color: var(--mi-text); letter-spacing: -.02em; }
    h1 { font-size: clamp(1.8rem, 3vw, 2.45rem); line-height: 1.12; }
    h2, h3 { font-size: clamp(1.1rem, 1.8vw, 1.4rem); }
    p, label, [data-testid="stCaptionContainer"] { overflow-wrap: anywhere; }
    [data-testid="stCaptionContainer"], small { color: var(--mi-muted); }
    a { color: #82b9f0 !important; }
    [data-testid="stTopNav"] {
      box-sizing: border-box;
      width: 100%;
      max-width: 1160px;
      margin-inline: auto;
      border-bottom: 1px solid var(--mi-border);
      background: var(--mi-bg);
    }
    [data-testid="stTopNavLink"] {
      color: var(--mi-muted) !important;
      font-weight: 600;
      border-radius: 9px 9px 0 0;
    }
    [data-testid="stTopNavLink"][aria-current="page"] {
      color: var(--mi-text) !important;
      background: var(--mi-surface-raised);
      box-shadow: inset 0 -2px 0 var(--mi-accent);
    }
    .st-key-mobile-navigation { display: none; }
    .st-key-mobile-navigation [data-testid="stPageLink"] a {
      box-sizing: border-box;
      justify-content: center;
      max-width: 100%;
      padding: .55rem .7rem;
      border: 1px solid var(--mi-border);
      border-radius: 9px;
      background: var(--mi-surface);
      color: var(--mi-muted) !important;
      font-weight: 650;
    }
    .st-key-mobile-navigation [data-testid="stPageLink"] a[aria-current="page"] {
      color: var(--mi-text) !important;
      background: var(--mi-accent-soft);
      border-color: #38618b;
    }
    .st-key-mobile-navigation .st-key-mobile-nav-predict-active [data-testid="stPageLink-NavLink"],
    .st-key-mobile-navigation .st-key-mobile-nav-research-active [data-testid="stPageLink-NavLink"] {
      color: var(--mi-text) !important;
      background: var(--mi-accent-soft);
      border-color: #38618b;
      box-shadow: inset 0 -2px 0 var(--mi-accent);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      box-sizing: border-box;
      max-width: 100%;
      border-color: var(--mi-border) !important;
      border-radius: 15px !important;
      background: var(--mi-surface);
    }
    [data-testid="stMetric"] {
      box-sizing: border-box;
      max-width: 100%;
      background: var(--mi-surface);
      border: 1px solid var(--mi-border);
      border-radius: 13px;
      padding: .9rem 1rem;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { color: var(--mi-muted); }
    [data-testid="stMetricValue"] { color: var(--mi-text); }
    [data-testid="stTextInput"], [data-testid="stSelectbox"],
    [data-testid="stTextArea"], [data-testid="stButton"] {
      min-width: 0;
      max-width: 100%;
    }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
      box-sizing: border-box;
      max-width: 100%;
      border-radius: 9px;
      background: #10161e;
      border-color: #354151;
      color: var(--mi-text);
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder { color: #778598; }
    [data-testid="stBaseButton-primary"] {
      max-width: 100%;
      border-radius: 9px;
      border-color: var(--mi-accent);
      background: var(--mi-accent);
      color: #07111c;
      font-weight: 700;
    }
    [data-testid="stBaseButton-primary"]:hover {
      border-color: #82b9f0;
      background: #82b9f0;
      color: #07111c;
    }
    [data-testid="stBaseButton-secondary"] {
      max-width: 100%;
      border-radius: 9px;
      border-color: #364456;
      background: #18212c;
      color: var(--mi-text);
    }
    [data-testid="stRadio"] [role="radiogroup"] {
      display: flex;
      flex-wrap: wrap;
      gap: .45rem;
    }
    [data-testid="stRadio"] label {
      box-sizing: border-box;
      border: 1px solid var(--mi-border);
      border-radius: 9px;
      background: #10161e;
      color: var(--mi-muted);
      padding: .55rem .8rem;
    }
    [data-testid="stRadio"] label:has(input:checked) {
      border-color: #38618b;
      background: var(--mi-accent-soft);
      color: var(--mi-text);
    }
    [data-testid="stHorizontalBlock"] { min-width: 0; }
    .mi-brand {
      margin: .15rem 0 .35rem;
      color: #90a2b8;
      font-size: .72rem;
      font-weight: 750;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .mi-brand span { color: var(--mi-accent); }
    .mi-page-subtitle { color: var(--mi-muted); margin-top: -.5rem; }
    .mi-result-card {
      box-sizing: border-box;
      width: 100%;
      min-width: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: .6rem;
      padding: .85rem;
      border: 1px solid var(--mi-border);
      border-radius: 14px;
      background: var(--mi-surface);
    }
    .mi-circle {
      width: clamp(104px, 10vw, 136px);
      aspect-ratio: 1;
      border-radius: 50%;
      display: grid;
      place-items: center;
      transition: background .3s ease;
    }
    .mi-circle-inner {
      box-sizing: border-box;
      width: calc(100% - 28px);
      aspect-ratio: 1;
      border-radius: 50%;
      background: #111720;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: .2rem;
    }
    .mi-circle-inner strong { color: var(--mi-text); font-size: clamp(1rem, 1.8vw, 1.4rem); }
    .mi-circle-inner span { color: var(--mi-muted); font-size: .72rem; text-align: center; }
    .mi-circle-label { color: var(--mi-text); font-weight: 700; text-align: center; }
    .mi-answer-panel { border-left: 3px solid var(--mi-accent); padding-left: .85rem; }
    @media (max-width: 760px) {
      .stMainBlockContainer, .block-container { padding-inline: .9rem !important; }
      .st-key-mobile-navigation { display: block; margin-bottom: .8rem; }
      .st-key-mobile-navigation [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 100px !important;
      }
      [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        align-items: stretch;
        gap: .75rem !important;
      }
      [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        box-sizing: border-box;
        flex: 1 1 170px !important;
        min-width: 0 !important;
        width: auto !important;
      }
      [data-testid="stTopNav"] { display: none; }
    }
    @media (max-width: 380px) {
      [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex-basis: 100% !important; }
      .mi-result-card { padding: .7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from ui_pages import render_predict_page, render_research_page  # noqa: E402


research = st.Page(
    lambda: _render_research_page(),
    title="Research Library",
    icon=":material/library_books:",
    url_path="research",
)


def _render_mobile_navigation(active_page: str) -> None:
    with st.container(key="mobile-navigation"):
        left, right = st.columns(2, gap="small")
        with left.container(key="mobile-nav-predict-active" if active_page == "predict" else "mobile-nav-predict"):
            st.page_link(predict, label="Predict", icon=":material/science:", width="stretch")
        with right.container(key="mobile-nav-research-active" if active_page == "research" else "mobile-nav-research"):
            st.page_link(research, label="Research Library", icon=":material/library_books:", width="stretch")


def _render_research_page() -> None:
    _render_mobile_navigation("research")
    render_research_page()


def _render_predict_page() -> None:
    _render_mobile_navigation("predict")
    render_predict_page(research)


predict = st.Page(
    _render_predict_page,
    title="Predict",
    icon=":material/science:",
    url_path="predict",
    default=True,
)

navigation = st.navigation([predict, research], position="top")
navigation.run()
