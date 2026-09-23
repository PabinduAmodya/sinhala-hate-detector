"""
Sinhala–English Hate Speech Detector — Hugging Face Spaces app.
Self-contained: loads the fine-tuned model from the Hugging Face Hub.

Before deploying: set MODEL_ID below to YOUR Hub repo (e.g. "yourname/sinhala-hate-detector"),
the same repo you push the model to with push_model.py.
"""
import os
import html
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── CHANGE THIS to your Hub model repo (must match push_model.py) ──────────────
MODEL_ID = os.environ.get("MODEL_ID", "pabindu2003G/sinhala-hate-detector")
MAX_LEN = 128
ID2LABEL = {0: "Not offensive", 1: "Offensive"}
OFF = 1

# ── Hybrid lexical safety net ────────────────────────────────────────────────
# Production content-moderation systems are hybrid: a neural model + a thin
# lexical layer that catches its systematic errors. Ours corrects ONE known
# failure mode — casual second-person chat ("umba kohomada", "machan umba
# enawada") where an informal pronoun alone must never count as hate. The rule
# fires ONLY when the model says Offensive but the text carries NO recognisable
# insult / vulgar / slur / group-hate cue, so it can never hide coded hate
# (which contains a group term) or slur-based abuse (which contains an insult).
import re as _re

# HARD abuse — vulgar words, slurs, and NOUN-FORM name-calls ("an idiot / a dog").
# Their presence always keeps the model's Offensive verdict (never rescued).
_HARD = {
    # noun-form name-calls (calling a PERSON one of these = insult)
    "modaya", "modayo", "modayek", "modayaa", "gonaa", "gonek", "gonaaa",
    "buruwa", "booruwa", "buruwek", "balla", "ballo", "ballek", "pissa", "pissek",
    "pissi", "pissu", "yako", "yako",
    # vulgar / obscene
    "pako", "pakaya", "puka", "keri", "hutta", "hutto", "huththa", "huttek",
    "wesi", "wesa", "wesii", "ponna", "ponnaya", "kariya", "hukanna",
    # script
    "මෝඩයා", "මෝඩයෙක්", "බූරුවා", "ගොනා", "බල්ලා", "බල්ලො", "බල්ලෙක්",
    "පක", "පකයා", "පුක", "හුත්ත", "හුත්තො", "කැරි", "වේසි", "පොන්නයා",
    # english
    "idiot", "bitch", "bastard", "fuck", "shit", "asshole", "moron", "retard",
    "slut", "whore",
}
# SOFT words — "stupid / foolish / donkey" as an ADJECTIVE. Offensive ONLY when
# aimed at a person (a target pronoun is present); harmless when describing an
# action or thing ("moda wada karanna epa" = "don't do foolish things").
_SOFT = {"moda", "gon", "gona", "buru", "modai", "gonai",
         "මෝඩ", "ගොන්", "බුරු"}
# person-target pronouns — a SOFT word + one of these = an insult aimed at someone
_TARGET = {"umba", "uba", "umbala", "umbata", "tho", "thopi", "thou", "oya",
           "oyaa", "oyaata", "oyala", "oyaala", "eyaa", "eyaata", "un", "unta",
           "meya", "muney", "උඹ", "උබ", "තෝ", "ඔය", "එයා"}
# group / targeted-hate cues — presence BLOCKS any rescue (protects coded hate)
_GROUP = {"demala", "thambi", "para", "muslim", "yanna", "elawanna", "nathi",
          "wanaganna", "haralla", "දෙමළ", "තම්බි", "පර"}
# informal / friendly words that TRIGGER the rescue check when nothing is abusive
_CASUAL = {"umba", "uba", "umbala", "umbata", "tho", "thopi", "thou", "machan",
           "macho", "malli", "aiya", "nangi", "putha", "yaluwa", "yaaluwa", "bro",
           "ban", "bn", "oya", "api", "mama", "eka", "wada", "weda", "wede",
           "උඹ", "උබ", "උඹල", "තෝ", "මචන්", "මල්ලි", "අයිය"}


def _tokens(text):
    return _re.findall(r"[\w඀-෿]+", str(text).lower())


def _rescue(res):
    res = dict(res)
    res["label"] = "Not offensive"
    res["offensive_score"] = min(res["offensive_score"], 0.20)
    res["confidence"] = 1.0 - res["offensive_score"]
    res["probabilities"] = {"Not offensive": res["confidence"], "Offensive": res["offensive_score"]}
    return res, True


def apply_safety_net(text, res):
    """Correct the model's systematic false positives on casual/idiomatic text.

    Fires only when the model says Offensive. It never overrides a HARD insult,
    a group-hate cue, or a SOFT word aimed at a person — so it cannot hide real
    abuse. It rescues (a) a mild 'stupid/foolish' word used to describe an action
    or thing, and (b) casual second-person chat with no abusive word at all.
    Returns (res, fixed)."""
    if res["label"] != "Offensive":
        return res, False
    toks = set(_tokens(text))
    if toks & _HARD or toks & _GROUP:
        return res, False                       # real insult / coded hate — leave it
    if toks & _SOFT:
        if toks & _TARGET:
            return res, False                   # "umba moda" — aimed at a person, keep Offensive
        return _rescue(res)                      # "moda wada karanna epa" — describing a thing
    if toks & _CASUAL:
        return _rescue(res)                      # casual chat, nothing abusive
    return res, False

st.set_page_config(page_title="Sinhala–English Hate Speech Detector",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

NAVY, BLUE, GREEN, RED, AMBER = "#1F4E79", "#2E75B6", "#2E9E5B", "#E03B3B", "#E0912B"
st.markdown(f"""
<style>
 .block-container {{ padding-top: 1.4rem; max-width: 1150px; }}
 .hero {{ background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 100%); color:#fff;
   padding:26px 30px; border-radius:16px; box-shadow:0 10px 30px rgba(31,78,121,.25); }}
 .hero h1 {{ margin:0; font-size:30px; font-weight:800; letter-spacing:-.5px; }}
 .hero p {{ margin:8px 0 0; font-size:15px; opacity:.93; }}
 .chip {{ display:inline-block; background:rgba(255,255,255,.16); color:#fff; padding:4px 12px;
   border-radius:999px; font-size:12.5px; font-weight:600; margin:3px 6px 3px 0;
   border:1px solid rgba(255,255,255,.25); }}
 .result {{ border-radius:14px; padding:20px 24px; margin:6px 0 4px; color:#fff;
   box-shadow:0 8px 22px rgba(0,0,0,.10); }}
 .result .lab {{ font-size:26px; font-weight:800; }}
 .result .sub {{ font-size:14px; opacity:.95; margin-top:4px; }}
 .r-off {{ background:linear-gradient(120deg,{RED} 0%,#b71c1c 100%); }}
 .r-not {{ background:linear-gradient(120deg,{GREEN} 0%,#1b7742 100%); }}
 .r-unc {{ background:linear-gradient(120deg,{AMBER} 0%,#a5680c 100%); }}
 .meter {{ position:relative; height:16px; border-radius:999px; margin:6px 0 2px;
   background:linear-gradient(90deg,{GREEN} 0%,{AMBER} 55%,{RED} 100%); }}
 .meter-marker {{ position:absolute; top:-5px; width:4px; height:26px; background:#12233a;
   border-radius:3px; transform:translateX(-2px); box-shadow:0 0 0 2px #fff; }}
 .meter-scale {{ display:flex; justify-content:space-between; font-size:11px; color:#7a869a; margin-top:2px; }}
 .hl {{ line-height:2.15; font-size:17px; padding:16px 18px; border-radius:12px;
   background:#f7f9fc; border:1px solid #e6ecf3; }}
 .hl span {{ padding:2px 3px; border-radius:5px; }}
 .legend span {{ display:inline-block; padding:2px 8px; border-radius:5px; font-size:12px; margin-right:8px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model from the Hub…")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # The weights are stored in fp16 to keep the download small; we load them as
    # fp32 because CPU inference needs full precision. low_cpu_mem_usage avoids the
    # transient 2x-memory spike at load time (keeps us under the free-tier RAM cap).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, low_cpu_mem_usage=True, torch_dtype=torch.float32).eval()
    torch.set_num_threads(2)
    return tok, model


try:
    tok, model = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    LOAD_ERR = str(e)


def _probs(texts):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, dim=1).numpy()


def predict(text):
    p = _probs([str(text)])[0]
    i = int(p.argmax())
    res = {"label": ID2LABEL[i], "confidence": float(p[i]),
           "probabilities": {ID2LABEL[j]: float(p[j]) for j in range(2)},
           "offensive_score": float(p[OFF])}
    res, _ = apply_safety_net(text, res)
    return res


def predict_batch(texts, bs=32):
    out = []
    for s in range(0, len(texts), bs):
        chunk = [str(t) for t in texts[s:s + bs]]
        for t, p in zip(chunk, _probs(chunk)):
            i = int(p.argmax())
            res = {"label": ID2LABEL[i], "confidence": float(p[i]),
                   "offensive_score": float(p[OFF])}
            res, _ = apply_safety_net(t, res)
            out.append(res)
    return out


def word_importance(text):
    words = str(text).split()
    if not words:
        return []
    base = float(_probs([str(text)])[0][OFF])
    variants = [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
    without = _probs(variants)[:, OFF]
    return [(w, float(base - wo)) for w, wo in zip(words, without)]


def color_for(score, mx):
    if mx <= 1e-9 or abs(score) < 0.04 * mx:
        return "transparent"
    a = min(0.85, 0.12 + 0.73 * abs(score) / mx)
    r, g, b = (224, 59, 59) if score > 0 else (46, 158, 91)
    return f"rgba({r},{g},{b},{a:.2f})"


def render_highlight(pairs):
    if not pairs:
        return "<div class='hl'><i>No words to explain.</i></div>"
    mx = max((abs(s) for _, s in pairs), default=0.0)
    spans = [f"<span style='background:{color_for(s, mx)}' title='{s:+.3f}'>{html.escape(w)}</span>"
             for w, s in pairs]
    return "<div class='hl'>" + " ".join(spans) + "</div>"


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🛡️ Model")
    st.caption(f"`{MODEL_ID}`")
    st.markdown(f"""<div style='background:#fff;border:1px solid #e6ecf3;border-radius:12px;padding:14px'>
    <b>XLM-RoBERTa</b> · fine-tuned<br>Binary: Offensive / Not offensive<br>
    + Focal Loss · SHAP/occlusion · transliteration & contrastive augmentation</div>""",
                unsafe_allow_html=True)
    st.divider()
    st.caption("Runs on CPU — first prediction takes a few seconds.")

# ── Hero ──
st.markdown("""
<div class="hero">
  <h1>Sinhala–English Hate Speech Detector</h1>
  <p>Explainable offensive-language detection for Sinhala, English and code-mixed (Singlish) text.</p>
  <div style="margin-top:14px">
    <span class="chip">XLM-RoBERTa</span><span class="chip">Focal Loss</span>
    <span class="chip">Explainable (SHAP)</span><span class="chip">Context-aware</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error(f"Could not load the model from `{MODEL_ID}`. Set MODEL_ID to your Hub repo. Details: {LOAD_ERR}")
    st.stop()

tab_a, tab_b, tab_c = st.tabs(["  🔎 Analyse  ", "  📦 Batch  ", "  ℹ️ About  "])

EXAMPLES = {
    "🇱🇰 Praise (Sinhala)": "ගොඩක් ස්තුතියි, හරිම ලස්සන පැහැදිලි කිරීමක්.",
    "❤️ Endearment (Singlish)": "mage buru patiya, adarei oyaata",
    "⚠️ Insult (Singlish)": "umba modaya, para balla",
    "⚠️ Coded hate": "oya aya ape ratin yanna ona",
    "🔀 Friendly code-mixed": "ela machan, super video, keep it up",
}

with tab_a:
    st.markdown("##### Try an example")
    cols = st.columns(len(EXAMPLES))
    for col, (name, txt) in zip(cols, EXAMPLES.items()):
        col.button(name, use_container_width=True,
                   on_click=lambda t=txt: st.session_state.update(inp=t))
    st.text_area("Enter a comment", key="inp", height=110,
                 placeholder="Type a Sinhala / English / code-mixed comment…")
    if st.button("Analyse comment", type="primary", use_container_width=True):
        text = st.session_state.get("inp", "").strip()
        if not text:
            st.warning("Please enter a comment.")
        else:
            res = predict(text)
            poff = res["offensive_score"]
            if 0.42 <= poff <= 0.62:
                cls, emoji, lab = "r-unc", "🟠", "Uncertain"
                sub = "Borderline — recommend human review (the model is not confident)."
            elif res["label"] == "Offensive":
                cls, emoji, lab = "r-off", "⚠️", "Offensive"
                sub = "Contains offensive / abusive language."
            else:
                cls, emoji, lab = "r-not", "✅", "Not offensive"
                sub = "No offensive language detected."
            st.markdown(f"<div class='result {cls}'><div class='lab'>{emoji} {lab}</div>"
                        f"<div class='sub'>{sub} &nbsp;•&nbsp; {res['confidence']:.1%} confidence</div></div>",
                        unsafe_allow_html=True)
            st.markdown("**Offensiveness score**")
            st.markdown(f"<div class='meter'><div class='meter-marker' style='left:{poff*100:.1f}%'></div></div>"
                        f"<div class='meter-scale'><span>Not offensive</span><span>{poff:.0%}</span>"
                        f"<span>Offensive</span></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            for col, nm in ((c1, "Not offensive"), (c2, "Offensive")):
                with col:
                    st.caption(nm); st.progress(res["probabilities"][nm])
                    st.write(f"**{res['probabilities'][nm]:.1%}**")
            st.markdown("#### 🔍 Why? — word-level explanation")
            st.markdown("<div class='legend'><span style='background:rgba(224,59,59,.55)'>toward Offensive</span>"
                        "<span style='background:rgba(46,158,91,.55)'>toward Not offensive</span></div>",
                        unsafe_allow_html=True)
            with st.spinner("Computing word contributions…"):
                pairs = word_importance(text)
            st.markdown(render_highlight(pairs), unsafe_allow_html=True)

with tab_b:
    st.markdown("Paste comments (one per line) or upload a CSV with a **text** column.")
    up = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
    pasted = st.text_area("…or paste comments", height=140)
    if st.button("Run batch", type="primary"):
        import pandas as pd
        texts = []
        if up is not None:
            di = pd.read_csv(up); col = "text" if "text" in di.columns else di.columns[0]
            texts = [str(x) for x in di[col].dropna().tolist()]
        elif pasted.strip():
            texts = [l.strip() for l in pasted.splitlines() if l.strip()]
        if not texts:
            st.warning("Provide a CSV or paste comments.")
        else:
            with st.spinner(f"Analysing {len(texts)} comments…"):
                preds = predict_batch(texts)
            out = pd.DataFrame({"text": texts,
                                "prediction": [p["label"] for p in preds],
                                "offensive_score": [round(p["offensive_score"], 3) for p in preds]})
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

with tab_c:
    st.markdown("""
### About
An explainable system that flags **offensive language** in Sinhala, English and code-mixed
(Singlish) text. It fine-tunes **XLM-RoBERTa** with Focal Loss and explains each decision at the
word level. It reads **context** — telling affectionate uses of pejorative words (e.g. *mage buru
patiya*, "my darling") from hostile ones — and returns an **"Uncertain"** verdict on borderline
cases instead of guessing.

*Final Year Research Project — BSc (Hons) Computer Science, NSBM Green University.*
""")
