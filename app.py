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

# ── Linguistic safety layer ──────────────────────────────────────────────────
# Threats (incl. pro-drop Sinhala threats with no pronoun), curses, identity-based hate,
# obscenity and name-calling — the harms the neural model is weakest on — plus narrow
# rescues of its known false positives. It lives in rules.py so it is tested offline
# (scripts/eval_suite.py, scripts/audit_youtube.py) without Streamlit.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import analyse, apply_safety_net, context_reason, evidence  # noqa: E402

RULE_NAMES = {
    "threat_target": "threat of violence", "threat_implied": "threat of violence (implied target)",
    "violence_call": "call for violence", "curse": "death wish / curse",
    "group_slur": "identity slur", "group_violence": "violence against a group",
    "group_expulsion": "expulsion of a group", "group_dehumanise": "dehumanising a group",
    "group_boycott": "hate trope / boycott call", "vulgar": "obscenity", "name_call": "name-calling",
    "endearment": "endearment", "literal_verb": "literal action verb", "soft_thing": "mild word about a thing",
    "figurative": "figurative 'dying' (of laughter, hunger...)",
    "negated": "negated threat", "counter_speech": "counter-speech", "supportive": "supportive statement",
    "festival": "religious/cultural observance (bias correction)",
    "casual": "casual / friendly chat",
}

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
 .hl {{ color:#12233a; line-height:2.15; font-size:17px; padding:16px 18px; border-radius:12px;
   background:#f7f9fc; border:1px solid #e6ecf3; }}
 .hl span {{ padding:2px 3px; border-radius:5px; }}
 .legend span {{ color:#12233a; display:inline-block; padding:2px 8px; border-radius:5px; font-size:12px; margin-right:8px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model from the Hub…")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # The weights are stored in fp16 to keep the download small; we load them as
    # fp32 because CPU inference needs full precision. low_cpu_mem_usage avoids the
    # transient 2x-memory spike at load time (keeps us under the free-tier RAM cap).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, low_cpu_mem_usage=True, dtype=torch.float32).eval()
    torch.set_num_threads(2)
    return tok, model


try:
    tok, model = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    LOAD_ERR = str(e)


# Decision threshold ships WITH the model (config.json "decision_threshold"), so the app can never pair a
# model with the wrong threshold. v3 = 0.614 (label-free calibration); older models fall back to 0.5.
THRESH = float(getattr(model.config, "decision_threshold", 0.5)) if MODEL_OK else 0.5
UNC_LO, UNC_HI = 0.42, max(0.62, THRESH + 0.01)      # "Uncertain — recommend human review" band


def _probs(texts):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, dim=1).numpy()


def predict(text):
    p = _probs([str(text)])[0]
    i = OFF if p[OFF] >= THRESH else 1 - OFF
    res = {"label": ID2LABEL[i], "confidence": float(p[i]),
           "probabilities": {ID2LABEL[j]: float(p[j]) for j in range(2)},
           "offensive_score": float(p[OFF]), "model_score": float(p[OFF])}
    sig = analyse(text)
    res, _ = apply_safety_net(text, res, sig)
    res["reason"] = context_reason(text, res, sig)
    res["evidence"] = evidence(text, res, sig)
    return res


def predict_batch(texts, bs=32):
    out = []
    for s in range(0, len(texts), bs):
        chunk = [str(t) for t in texts[s:s + bs]]
        for t, p in zip(chunk, _probs(chunk)):
            i = OFF if p[OFF] >= THRESH else 1 - OFF
            res = {"label": ID2LABEL[i], "confidence": float(p[i]),
                   "offensive_score": float(p[OFF]), "model_score": float(p[OFF])}
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
    st.markdown(f"""<div style='background:#fff;color:#20334a;border:1px solid #e6ecf3;border-radius:12px;padding:14px'>
    <b>XLM-RoBERTa</b> · fine-tuned<br>Binary: Offensive / Not offensive<br>
    class-weighted fine-tuning · transliteration & contrastive augmentation<br>
    + linguistic safety layer (threats, curses, identity hate)<br>
    explanations: word-level occlusion</div>""",
                unsafe_allow_html=True)
    st.divider()
    st.caption("Runs on CPU — first prediction takes a few seconds.")

# ── Hero ──
st.markdown("""
<div class="hero">
  <h1>Sinhala–English Hate Speech Detector</h1>
  <p>Explainable offensive-language detection for Sinhala, English and code-mixed (Singlish) text.</p>
  <div style="margin-top:14px">
    <span class="chip">XLM-RoBERTa</span><span class="chip">Context-aware</span>
    <span class="chip">Threat &amp; hate safety layer</span><span class="chip">Explainable</span>
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
    "⚠️ Threat (no pronoun)": "කපල මරන්නේ දැන ගනින්",
    "⚠️ Coded hate": "oya aya ape ratin yanna ona",
    "🤝 Friendly slang": "bosa machan uba kohomada, ela video bro",
}

with tab_a:
    st.markdown("##### Try an example")
    cols = st.columns(len(EXAMPLES))
    for col, (name, txt) in zip(cols, EXAMPLES.items()):
        col.button(name, width="stretch",
                   on_click=lambda t=txt: st.session_state.update(inp=t))
    st.text_area("Enter a comment", key="inp", height=110,
                 placeholder="Type a Sinhala / English / code-mixed comment…")
    if st.button("Analyse comment", type="primary", width="stretch"):
        text = st.session_state.get("inp", "").strip()
        if not text:
            st.warning("Please enter a comment.")
        else:
            res = predict(text)
            poff = res["offensive_score"]
            if not res.get("rule") and UNC_LO <= poff <= UNC_HI:
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
            st.markdown("#### 🧠 Why? — context reasoning")
            st.markdown(
                f"<div style='background:#eef4fb;border:1px solid #cfe0f3;border-left:5px solid {BLUE};"
                f"border-radius:10px;padding:14px 16px;font-size:15.5px;color:#20334a'>"
                f"{html.escape(res['reason'])}</div>",
                unsafe_allow_html=True)
            if res.get("rule"):
                chips = "".join(f"<span style='background:#fde8c8;color:#5a3a00;border:1px solid #f0c27a;border-radius:6px;"
                                f"padding:2px 8px;margin:2px 4px 2px 0;display:inline-block'>{html.escape(w)}</span>"
                                for w in res.get("evidence", []))
                st.markdown(
                    f"<div style='margin-top:8px;font-size:14px;color:#20334a'>🛡️ <b>Decided by the linguistic "
                    f"safety layer</b> — {html.escape(RULE_NAMES.get(res['rule'], res['rule']))}. "
                    f"Neural model score alone: {res['model_score']:.0%}."
                    + (f"<br>Trigger words: {chips}" if chips else "") + "</div>",
                    unsafe_allow_html=True)
            with st.expander("Show word-level signals from the neural model (its raw score, before the safety layer)"):
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
                                "offensive_score": [round(p["offensive_score"], 3) for p in preds],
                                "model_score": [round(p["model_score"], 3) for p in preds],
                                "decided_by": [RULE_NAMES.get(p.get("rule"), "neural model") for p in preds]})
            st.dataframe(out, width="stretch", hide_index=True)
            st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

with tab_c:
    st.markdown("""
### About
An explainable **hybrid** system that flags **offensive language** in Sinhala, English and
code-mixed (Singlish) text. A fine-tuned **XLM-RoBERTa** model (SOLD dataset + transliteration and
contrastive augmentation, class-weighted fine-tuning) reads the language, and a transparent
**linguistic safety layer** catches what the model is blind to and explains each decision.

**What the safety layer catches (the model alone misses most of these):**
- **Threats**, including Sinhala *pro-drop* threats with no pronoun — *කපල මරන්නේ දැන ගනින්*
  ("I'll cut and kill (you), know that") — recognised from the verb's intent form (-නවා, -න්නම්, -න්නේ),
  intimidation markers (*දැනගනින්*, *බලාගනින්*) and serial verbs (*ගහලා මරනවා*).
- **Curses / death wishes** (*maren yako*, *umbata hena gahanna*), **identity-based hate** (slurs such as
  *hambaya*, calls to kill or expel a group, dehumanisation, boycott tropes), obscenity and name-calling.
- It does **not** fire on literal verbs (*cake eka kapamu*, *current eka kapanawa*), news reports,
  negated threats (*maranne na*), counter-speech, condolences (*මරණයට සංවේදනා*) or slang (*maru* = awesome).

**How it reads context — not keywords:**
- Tells an affectionate pejorative (*mage buru patiya*, "my darling") from a hostile one (*umba buruwa*).
- Knows a mild word describing a *thing* (*moda wada* — "foolish work") isn't a personal insult.
- Treats informal address (*umba*, *machan*, *bosa*) as friendly banter unless it targets a person.
- Flags **coded hate** that has no slur (e.g. calls to expel a group), and abstains with an
  **"Uncertain"** verdict on genuinely borderline text.
- Normalises **romanized-Sinhala spelling** (*hutta / huththa / huttoo* → one form), the challenge
  the literature identifies as hardest for this task.

This model + explainable-lexicon design follows validated work on lexicon-enhanced transformers for
low-resource hate-speech detection.

*Final Year Research Project — BSc (Hons) Computer Science, NSBM Green University.*
""")
