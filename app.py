import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# --------------------------------------------------

st.set_page_config(
    page_title="CineSentiment – Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# IMDb-INSPIRED THEME  (amber / charcoal / black)
# --------------------------------------------------

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:ital,wght@0,400;0,600;0,700;1,400&display=swap');

/* ── Root palette (IMDb: #F5C518 gold, #121212 bg, #1F1F1F card) ── */
:root {
    --gold:      #F5C518;
    --gold-dark: #D4A800;
    --bg:        #121212;
    --card:      #1F1F1F;
    --card2:     #2A2A2A;
    --border:    #3A3A3A;
    --text:      #E8E8E8;
    --muted:     #9E9E9E;
    --positive:  #4CAF50;
    --negative:  #F44336;
    --radius:    10px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Barlow', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%);
    border-bottom: 3px solid var(--gold);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🎬';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.07;
    pointer-events: none;
}
.hero-badge {
    background: var(--gold);
    color: #000;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    padding: 0.35rem 1rem;
    border-radius: 6px;
    letter-spacing: 2px;
    flex-shrink: 0;
}
.hero-title  { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem;
               letter-spacing: 2px; color: #fff; margin: 0; }
.hero-sub    { color: var(--muted); font-size: 0.95rem; margin-top: 0.3rem; }

/* ── Section headings ── */
.section-head {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 2px;
    color: var(--gold);
    border-left: 4px solid var(--gold);
    padding-left: 0.7rem;
    margin: 2rem 0 1rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-highlight {
    background: var(--card2);
    border: 2px solid var(--gold);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    margin-bottom: 1rem;
}

/* ── Model radio pills ── */
div[data-testid="stRadio"] > label {
    color: var(--muted);
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
div[data-testid="stRadio"] > div {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label > div:first-child {
    display: none;          /* hide default radio dot */
}
div[data-testid="stRadio"] label {
    background: var(--card2);
    border: 2px solid var(--border);
    border-radius: 50px;
    padding: 0.45rem 1.3rem !important;
    cursor: pointer;
    transition: all .2s;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}
div[data-testid="stRadio"] label:hover {
    border-color: var(--gold);
    color: var(--gold) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked),
div[data-testid="stRadio"] label:has(> div > input:checked) {
    background: var(--gold);
    border-color: var(--gold);
    color: #000 !important;
}

/* ── Text area ── */
textarea {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    transition: border-color .25s;
}
textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,197,24,.15) !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button {
    background: var(--gold) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 2.4rem !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.15rem !important;
    letter-spacing: 2px !important;
    cursor: pointer;
    transition: background .2s, transform .1s;
    width: auto !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--gold-dark) !important;
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button:active { transform: translateY(0); }

/* ── Sample review cards ── */
.sample-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.9rem;
    margin-bottom: 1rem;
}
.sample-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    cursor: pointer;
    transition: border-color .2s, transform .15s;
    position: relative;
}
.sample-card:hover {
    border-color: var(--gold);
    transform: translateY(-2px);
}
.sample-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    display: inline-block;
}
.label-pos { background: rgba(76,175,80,.18); color: #4CAF50; }
.label-neg { background: rgba(244,67,54,.18); color: #F44336; }
.label-mid { background: rgba(245,197,24,.15); color: var(--gold); }
.sample-text { font-size: 0.88rem; color: var(--muted); line-height: 1.5; }

/* ── Result banner ── */
.result-pos {
    background: linear-gradient(90deg, rgba(76,175,80,.18) 0%, var(--card) 100%);
    border-left: 5px solid var(--positive);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
}
.result-neg {
    background: linear-gradient(90deg, rgba(244,67,54,.18) 0%, var(--card) 100%);
    border-left: 5px solid var(--negative);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
}
.result-emoji { font-size: 3rem; }
.result-label { font-family: 'Bebas Neue', sans-serif; font-size: 2rem;
                letter-spacing: 2px; }
.result-conf  { font-size: 0.95rem; color: var(--muted); }

/* ── Metric tiles ── */
div[data-testid="stMetric"] {
    background: var(--card2) !important;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem !important;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    font-family: 'Barlow', sans-serif;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Star rating display ── */
.stars { color: var(--gold); font-size: 1.2rem; letter-spacing: 2px; }

/* ── Tooltip chip ── */
.chip {
    display: inline-block;
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    color: var(--muted);
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MAX_LEN = 509

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    rnn  = load_model("simple_rnn_model.h5")
    lstm = load_model("lstm_model.h5")
    gru  = load_model("gru_model.h5")
    return rnn, lstm, gru


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


rnn_model, lstm_model, gru_model = load_models()
tokenizer = load_tokenizer()

# --------------------------------------------------
# PREPROCESS / PREDICT
# --------------------------------------------------

def preprocess_review(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded   = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded


def predict(model, review):
    x    = preprocess_review(review)
    prob = model.predict(x, verbose=0)[0][0]
    sentiment  = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
    return sentiment, confidence, prob


# --------------------------------------------------
# PLOTLY THEME HELPER
# --------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Barlow, sans-serif', color='#E8E8E8'),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(showgrid=False, color='#9E9E9E'),
    yaxis=dict(showgrid=True,  gridcolor='#2A2A2A', color='#9E9E9E'),
)

# --------------------------------------------------
# HERO BANNER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-badge">CS</div>
    <div>
        <div class="hero-title">CineSentiment Analyzer</div>
        <div class="hero-sub">Deep Learning · SimpleRNN · LSTM · GRU · Trained on IMDb 50 k</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SAMPLE REVIEWS
# --------------------------------------------------

SAMPLES = [
    {
        "label": "Positive",
        "genre": "Action",
        "stars": "★★★★★",
        "text": "An absolute masterpiece of modern cinema. The action sequences were breathtaking, the performances were Oscar-worthy, and the story had me on the edge of my seat from start to finish. I haven't felt this exhilarated by a film in years.",
    },
    {
        "label": "Positive",
        "genre": "Drama",
        "stars": "★★★★½",
        "text": "Beautifully crafted and emotionally resonant. The director strikes the perfect balance between heartbreak and hope. The lead actress delivers what might be the performance of the decade. Don't miss this one.",
    },
    {
        "label": "Negative",
        "genre": "Horror",
        "stars": "★½☆☆☆",
        "text": "Painfully predictable and riddled with plot holes. The jump scares were telegraphed a mile away, the characters made bafflingly stupid decisions, and the twist ending felt like it was written in five minutes. Total waste of two hours.",
    },
    {
        "label": "Negative",
        "genre": "Sci-Fi",
        "stars": "★★☆☆☆",
        "text": "Despite a promising premise and impressive visual effects, the film collapses under the weight of its incoherent screenplay. The dialogue is clunky, motivations are never established, and the third act abandons all logic.",
    },
    {
        "label": "Mixed",
        "genre": "Comedy",
        "stars": "★★★☆☆",
        "text": "Has its moments of genuine wit but ultimately feels uneven. The first half sparkles with sharp observations about modern life, yet the film loses its nerve in the finale and resorts to tired clichés. Worth a watch, just temper expectations.",
    },
    {
        "label": "Positive",
        "genre": "Thriller",
        "stars": "★★★★☆",
        "text": "A tightly wound thriller that never lets you breathe. The pacing is relentless, the twists are genuinely surprising, and the cinematography creates an atmosphere of dread that lingers long after the credits roll.",
    },
]

st.markdown('<div class="section-head">SAMPLE REVIEWS — CLICK TO LOAD</div>', unsafe_allow_html=True)

cols = st.columns(3)
for idx, s in enumerate(SAMPLES):
    label_cls = {"Positive": "label-pos", "Negative": "label-neg", "Mixed": "label-mid"}[s["label"]]
    with cols[idx % 3]:
        if st.button(
            f"{s['label']} · {s['genre']}  {s['stars']}",
            key=f"sample_{idx}",
            help=s["text"][:80] + "…",
        ):
            st.session_state["loaded_review"] = s["text"]
        st.markdown(
            f'<div class="sample-card">'
            f'<span class="sample-label {label_cls}">{s["label"]}</span> '
            f'<span class="chip">{s["genre"]}</span>'
            f'<div style="margin-top:0.1rem; color:var(--gold); font-size:0.85rem">{s["stars"]}</div>'
            f'<div class="sample-text">{s["text"][:110]}…</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# --------------------------------------------------
# CONTROLS
# --------------------------------------------------

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div class="section-head">YOUR REVIEW</div>', unsafe_allow_html=True)
    default_text = st.session_state.get("loaded_review", "")
    review = st.text_area(
        "Write or paste a movie review below:",
        value=default_text,
        height=200,
        placeholder="e.g. 'This film was an unforgettable journey through...'",
        label_visibility="collapsed",
    )
    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        analyze_clicked = st.button("🎬  ANALYZE SENTIMENT", use_container_width=True)
    with col_clear:
        if st.button("✕  CLEAR", use_container_width=True):
            st.session_state["loaded_review"] = ""
            st.rerun()

with right:
    st.markdown('<div class="section-head">SELECT MODEL</div>', unsafe_allow_html=True)
    selected_model = st.radio(
        "Model",
        ["SimpleRNN", "LSTM", "GRU"],
        label_visibility="collapsed",
    )
    st.markdown("""
    <div class="card" style="margin-top:1rem">
        <div style="font-size:0.82rem; color:var(--muted); line-height:1.7">
            <b style="color:var(--gold)">SimpleRNN</b> — Lightweight recurrent network.<br>
            <b style="color:var(--gold)">LSTM</b> — Long Short-Term Memory, handles long-range dependencies.<br>
            <b style="color:var(--gold)">GRU</b> — Gated Recurrent Unit, faster LSTM variant.<br><br>
            <span style="color:#666">Trained on the IMDb Large Movie Review Dataset (50 000 reviews).</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze_clicked:

    if not review.strip():
        st.warning("⚠️  Please enter a review or load one of the samples above.")
    else:
        model_dict = {"SimpleRNN": rnn_model, "LSTM": lstm_model, "GRU": gru_model}
        model = model_dict[selected_model]

        with st.spinner("Analyzing sentiment…"):
            sentiment, confidence, prob = predict(model, review)

        positive_prob = prob * 100
        negative_prob = (1 - prob) * 100

        # ── Result banner ──
        st.markdown('<div class="section-head">VERDICT</div>', unsafe_allow_html=True)
        if sentiment == "Positive":
            emoji, cls, col = "😍", "result-pos", "#4CAF50"
        else:
            emoji, cls, col = "😞", "result-neg", "#F44336"

        st.markdown(f"""
        <div class="{cls}">
            <table style="width:100%;border:none"><tr>
                <td style="font-size:3.5rem;width:70px;vertical-align:middle">{emoji}</td>
                <td style="vertical-align:middle">
                    <div class="result-label" style="color:{col}">{sentiment.upper()}</div>
                    <div class="result-conf">
                        <b style="color:var(--gold)">{confidence:.1f}%</b> confidence
                        &nbsp;·&nbsp; Model: <b>{selected_model}</b>
                        &nbsp;·&nbsp; Raw score: <b>{prob:.4f}</b>
                    </div>
                </td>
            </tr></table>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability breakdown ──
        st.markdown('<div class="section-head">PROBABILITY BREAKDOWN</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Positive Probability", f"{positive_prob:.2f}%")
        with m2:
            st.metric("Negative Probability", f"{negative_prob:.2f}%")

        # ── Single-model bar ──
        fig = go.Figure()
        fig.add_bar(
            x=["Positive", "Negative"],
            y=[positive_prob, negative_prob],
            marker_color=["#4CAF50", "#F44336"],
            text=[f"{positive_prob:.1f}%", f"{negative_prob:.1f}%"],
            textposition="outside",
        )
        fig.update_layout(
            title=f"Confidence Distribution — {selected_model}",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 110],
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── All-model comparison ──
        st.markdown("---")
        st.markdown('<div class="section-head">ALL MODELS COMPARISON</div>', unsafe_allow_html=True)

        with st.spinner("Running all three models…"):
            rnn_sent,  rnn_conf,  rnn_prob  = predict(rnn_model,  review)
            lstm_sent, lstm_conf, lstm_prob = predict(lstm_model, review)
            gru_sent,  gru_conf,  gru_prob  = predict(gru_model,  review)

        def badge(s):
            return "🟢 Positive" if s == "Positive" else "🔴 Negative"

        comparison_df = pd.DataFrame({
            "Model":          ["SimpleRNN", "LSTM", "GRU"],
            "Verdict":        [badge(rnn_sent), badge(lstm_sent), badge(gru_sent)],
            "Confidence (%)": [round(rnn_conf, 2), round(lstm_conf, 2), round(gru_conf, 2)],
            "Raw Score":      [round(rnn_prob, 4), round(lstm_prob, 4), round(gru_prob, 4)],
        })

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # ── Grouped bar (positive vs negative per model) ──
        models = ["SimpleRNN", "LSTM", "GRU"]
        pos_probs = [rnn_prob * 100,  lstm_prob * 100,  gru_prob * 100]
        neg_probs = [(1 - rnn_prob) * 100, (1 - lstm_prob) * 100, (1 - gru_prob) * 100]
        confs     = [rnn_conf, lstm_conf, gru_conf]

        fig2 = go.Figure()
        fig2.add_bar(name="Positive", x=models, y=pos_probs,
                     marker_color="#4CAF50",
                     text=[f"{v:.1f}%" for v in pos_probs], textposition="outside")
        fig2.add_bar(name="Negative", x=models, y=neg_probs,
                     marker_color="#F44336",
                     text=[f"{v:.1f}%" for v in neg_probs], textposition="outside")
        fig2.update_layout(
            title="Positive vs Negative Probability — All Models",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 115],
            barmode="group",
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#3A3A3A', borderwidth=1),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Confidence gauge (radial) ──
        st.markdown('<div class="section-head">MODEL CONFIDENCE GAUGES</div>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)

        def make_gauge(value, title, sentiment_lbl):
            color = "#4CAF50" if sentiment_lbl == "Positive" else "#F44336"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number=dict(suffix="%", font=dict(color=color, size=28)),
                title=dict(text=title, font=dict(color="#E8E8E8", size=14)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#555",
                               tickfont=dict(color="#777"), nticks=5),
                    bar=dict(color=color),
                    bgcolor="#1F1F1F",
                    bordercolor="#3A3A3A",
                    steps=[
                        dict(range=[0, 50],  color="#2A1A1A"),
                        dict(range=[50, 100], color="#1A2A1A"),
                    ],
                    threshold=dict(line=dict(color=color, width=3), value=value),
                )
            ))
            fig_g.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Barlow', color='#E8E8E8'),
                margin=dict(l=15, r=15, t=55, b=15),
                height=220,
            )
            return fig_g

        with g1:
            st.plotly_chart(make_gauge(rnn_conf,  "SimpleRNN", rnn_sent),  use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge(lstm_conf, "LSTM",      lstm_sent), use_container_width=True)
        with g3:
            st.plotly_chart(make_gauge(gru_conf,  "GRU",       gru_sent),  use_container_width=True)

        # ── Radar chart ──
        st.markdown('<div class="section-head">SENTIMENT RADAR</div>', unsafe_allow_html=True)
        cats = ["SimpleRNN Pos", "LSTM Pos", "GRU Pos",
                "GRU Neg",       "LSTM Neg", "SimpleRNN Neg"]
        vals = [rnn_prob*100, lstm_prob*100, gru_prob*100,
                (1-gru_prob)*100, (1-lstm_prob)*100, (1-rnn_prob)*100]
        vals.append(vals[0])
        cats.append(cats[0])

        fig3 = go.Figure()
        fig3.add_scatterpolar(
            r=vals, theta=cats,
            fill='toself',
            fillcolor='rgba(245,197,24,0.12)',
            line=dict(color='#F5C518', width=2),
            name='All Models',
        )
        fig3.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(range=[0, 100], tickfont=dict(color='#777'), gridcolor='#2A2A2A'),
                angularaxis=dict(tickfont=dict(color='#9E9E9E', size=11), gridcolor='#2A2A2A'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Barlow', color='#E8E8E8'),
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=30),
            height=360,
        )
        st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:var(--muted); font-size:0.82rem; padding: 1rem 0 0.5rem">
    CineSentiment &nbsp;·&nbsp; SimpleRNN · LSTM · GRU
    &nbsp;·&nbsp; IMDb 50 k dataset
    &nbsp;·&nbsp; Built with TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)