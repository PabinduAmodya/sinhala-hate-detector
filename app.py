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
import importlib as _importlib
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules as _rules  # noqa: E402
# Streamlit re-runs this script on every interaction and after a redeploy, but keeps imported modules
# cached: without a reload, a pushed rules.py is ignored until the server restarts (and a new app.py
# importing new names from the OLD cached module fails with ImportError).
_rules = _importlib.reload(_rules)
from rules import analyse, apply_safety_net, context_reason, evidence, light_normalize, profile, tokenize  # noqa: E402

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
    "emoji_threat": "weapon emoji aimed at a person", "mention": "word mentioned, not used",
    "rude_address": "contemptuous address (තොගේ / තොපේ)", "short_insult": "one-word insult",
    "short_neutral": "short text without abusive content", "play_context": "gaming / sports banter",
    "predicate_insult": "insult in what is said about the person", "register_benign": "rude-casual pronoun, harmless meaning",
    "sexual_harassment": "sexual harassment",
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
 .hl span.trig {{ outline:2px solid #12233a; outline-offset:1px; font-weight:700; }}
 /* every custom box sets its own background AND text colour, so it stays readable in dark mode */
 .box {{ border-radius:10px; padding:14px 16px; font-size:15px; color:#20334a; background:#eef4fb;
   border:1px solid #cfe0f3; margin:6px 0; }}
 .box-why {{ border-left:5px solid {BLUE}; font-size:15.5px; }}
 .box-rule {{ background:#fff7ea; border:1px solid #f0c27a; border-left:5px solid {AMBER}; color:#3d2a05; }}
 .box-cf {{ background:#f3f0fb; border:1px solid #d6cdf0; border-left:5px solid #6b4fbb; color:#261a4a; }}
 .trigchip {{ background:#fde8c8; color:#5a3a00; border:1px solid #f0c27a; border-radius:6px;
   padding:2px 8px; margin:2px 4px 2px 0; display:inline-block; }}
 .prof {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:6px 0 4px; }}
 .prof div {{ background:#f7f9fc; color:#12233a; border:1px solid #e6ecf3; border-radius:10px; padding:10px 12px; }}
 .prof small {{ display:block; color:#5d6b80; font-size:11.5px; text-transform:uppercase; letter-spacing:.4px; }}
 .prof b {{ font-size:15px; }}
 .scale {{ display:flex; gap:3px; margin-top:6px; }}
 .scale i {{ flex:1; height:7px; border-radius:3px; background:#dde4ee; }}
 .trace {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; font-size:14px; margin:4px 0 2px; }}
 .trace span {{ background:#f7f9fc; color:#12233a; border:1px solid #e6ecf3; border-radius:8px; padding:6px 10px; }}
 .trace em {{ color:#7a869a; font-style:normal; }}
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
    # the model sees evasion-normalised text (invisible characters, look-alike letters, s p a c e d words);
    # measured neutral on every benchmark and 2x more robust to homoglyphs (EVALUATION.md, round 3)
    texts = [light_normalize(t) for t in texts]
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
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
    res["profile"] = profile(text, res, sig)
    return res


def predict_batch(texts, bs=32):
    out = []
    for s in range(0, len(texts), bs):
        chunk = [str(t) for t in texts[s:s + bs]]
        for t, p in zip(chunk, _probs(chunk)):
            i = OFF if p[OFF] >= THRESH else 1 - OFF
            res = {"label": ID2LABEL[i], "confidence": float(p[i]),
                   "offensive_score": float(p[OFF]), "model_score": float(p[OFF])}
            sig = analyse(t)
            res, _ = apply_safety_net(t, res, sig)
            res["profile"] = profile(t, res, sig)
            out.append(res)
    return out


MAX_OCCLUSIONS = 40   # upper bound on model runs for one word-level explanation (most comments < 40 words)
XAI_METHOD = "delete"  # best of delete / mask / hybrid / SHAP / grad x input on SOLD human rationales
                       # (scripts/eval_xai.py, 499 unseen posts: AUPRC .733, comprehensiveness .521)


def word_importance(text, method=XAI_METHOD):
    """Occlusion attribution per word: p(full) - p(text with the word hidden).
    'delete' drops the word; 'mask' replaces it with XLM-R's <mask> token. Delete agreed slightly better
    with human rationales and was more faithful (EVALUATION.md, round 3), so it is the default."""
    words = str(text).split()
    if not words:
        return []
    # the model reads only the first MAX_LEN sub-word tokens: hiding a word beyond that cannot change its score,
    # so only the words it actually reads are attributed (the rest get 0) ...
    n_read, used = 0, 2
    for w in words:
        used += len(tok.tokenize(light_normalize(w))) or 1
        if used > MAX_LEN:
            break
        n_read += 1
    n_read = max(n_read, 1)
    # ... and a long comment is occluded in small groups of words, keeping the cost to <= MAX_OCCLUSIONS model runs
    k = max(1, -(-n_read // MAX_OCCLUSIONS))
    spans = [(a, min(a + k, n_read)) for a in range(0, n_read, k)]
    fill = [tok.mask_token] if method == "mask" else []
    variants = [" ".join(words[:a] + fill + words[b:]) or "." for a, b in spans]
    p = _probs([str(text)] + variants)[:, OFF]
    score = [0.0] * len(words)
    for (a, b), pv in zip(spans, p[1:]):
        for i in range(a, b):
            score[i] = float(p[0] - pv)
    return list(zip(words, score))


def trigger_positions(words, ev):
    """Indices of whitespace words that contain a safety-layer trigger word."""
    evs = {e.lower() for e in ev}
    return {i for i, w in enumerate(words) if set(tokenize(w)[0]) & evs or any(e in w for e in evs if not e.isalnum())}


def counterfactual(text, pairs, ev, max_edits=4):
    """Smallest greedy edit that flips the FULL system (model + safety layer) to Not offensive:
    hide the safety layer's trigger words first, then the words that push the model hardest.
    A contrastive explanation ("it is offensive BECAUSE of these words"), after Wachter et al. (2017)."""
    words = str(text).split()
    trig = trigger_positions(words, ev)
    order = sorted(trig, key=lambda i: -pairs[i][1]) + \
        [i for i, (_, s) in sorted(enumerate(pairs), key=lambda x: -x[1][1]) if s > 0 and i not in trig]
    removed = []
    for i in order[:max_edits]:
        removed.append(i)
        kept = [w for j, w in enumerate(words) if j not in removed]
        r = predict(" ".join(kept) or ".")
        if r["label"] != "Offensive":
            return [words[j] for j in sorted(removed)], r["offensive_score"]
    return None, None


def color_for(score, mx):
    if mx <= 1e-9 or abs(score) < 0.04 * mx:
        return "transparent"
    a = min(0.85, 0.12 + 0.73 * abs(score) / mx)
    r, g, b = (224, 59, 59) if score > 0 else (46, 158, 91)
    return f"rgba({r},{g},{b},{a:.2f})"


def render_highlight(pairs, trig=()):
    if not pairs:
        return "<div class='hl'><i>No words to explain.</i></div>"
    mx = max((abs(s) for _, s in pairs), default=0.0)
    spans = [f"<span class='{'trig' if i in trig else ''}' style='background:{color_for(s, mx)}' "
             f"title='model contribution {s:+.3f}'>{html.escape(w)}</span>"
             for i, (w, s) in enumerate(pairs)]
    return "<div class='hl'>" + " ".join(spans) + "</div>"


def render_profile(pr):
    lvl = pr["intensity"]
    col = [GREEN, "#8bbf4a", AMBER, "#e0702b", RED, "#b71c1c"]
    bars = "".join(f"<i style='background:{col[k] if k < lvl else '#dde4ee'}'></i>" for k in range(6))
    return (f"<div class='prof'>"
            f"<div><small>Category</small><b>{html.escape(pr['category'])}</b></div>"
            f"<div><small>Who is targeted</small><b>{html.escape(pr['target'])}</b></div>"
            f"<div><small>Severity</small><b>{lvl}/6 · {html.escape(pr['intensity_label'])}</b>"
            f"<div class='scale'>{bars}</div></div>"
            f"</div>")


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🛡️ Model")
    st.caption(f"`{MODEL_ID}`")
    st.markdown(f"""<div style='background:#fff;color:#20334a;border:1px solid #e6ecf3;border-radius:12px;padding:14px'>
    <b>XLM-RoBERTa</b> · fine-tuned<br>Binary: Offensive / Not offensive<br>
    class-weighted fine-tuning · transliteration & contrastive augmentation<br>
    + linguistic safety layer (threats, curses, identity hate)<br>
    explanations: offence profile, key words, counterfactual, word contributions</div>""",
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

EXAMPLES = {   # realistic social-media comments (label -> comment)
    "👍 Praise": "Supiri video ekak machan, next part eka ikmanata danna 🔥",
    "🏏 Cricket banter": "හෙට ක්‍රිකට් මැච් එකේදි අපි උඹලව කුඩු කරනවා 😂🏏",
    "💬 Casual pronoun": "meki mara lassanai neda? 😍",
    "🗣️ Civil criticism": "Me minister ge katha eka boru, data eka waradi. Evidence denna",
    "✋ Counter-speech": "Racism is wrong. Muslim shops boycott karanna kiyana post share karanna epa",
    "⚠️ Insult": "umba wage modayek nam dakala na, oluwe mola na",
    "⚠️ Threat": "උඹව ගෙදර ඇවිත් කපනවා බලාගෙන හිටපන්",
    "⚠️ Harassment": "Meki baduwak, number eka denna 😏",
    "⚠️ Hate trope": "Muslim kadawalin badu ganna epa wanda pethi danawa",
}

with tab_a:
    st.markdown("##### Try an example")
    items = list(EXAMPLES.items())
    for row in range(0, len(items), 3):
        for col, (name, txt) in zip(st.columns(3), items[row:row + 3]):
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
                sub = "Borderline — recommend human review."
            elif res["label"] == "Offensive":
                cls, emoji, lab = "r-off", "⚠️", "Offensive"
                sub = "Contains offensive / abusive language."
            else:
                cls, emoji, lab = "r-not", "✅", "Not offensive"
                sub = "No offensive language detected."
            st.markdown(f"<div class='result {cls}'><div class='lab'>{emoji} {lab}</div>"
                        f"<div class='sub'>{sub}</div></div>",
                        unsafe_allow_html=True)
            st.markdown("**Offensiveness score**")
            st.markdown(f"<div class='meter'><div class='meter-marker' style='left:{poff*100:.1f}%'></div></div>"
                        f"<div class='meter-scale'><span>Not offensive</span><span>{poff:.0%}</span>"
                        f"<span>Offensive</span></div>", unsafe_allow_html=True)
            st.markdown("#### 🧠 Why? — explanation")
            st.markdown(render_profile(res["profile"]), unsafe_allow_html=True)
            st.markdown(f"<div class='box box-why'>{html.escape(res['reason'])}</div>", unsafe_allow_html=True)
            if res.get("rule") and res.get("evidence"):
                # show the words exactly as the user typed them ("mrnw", not the internally restored "maranawa")
                typed = text.split()
                shown = [typed[i].strip(",.!?;:") for i in sorted(trigger_positions(typed, res["evidence"]))]
                chips = "".join(f"<span class='trigchip'>{html.escape(w)}</span>" for w in (shown or res["evidence"]))
                st.markdown(f"<div class='box box-rule'>🔑 <b>Key words</b> behind this decision: {chips}</div>",
                            unsafe_allow_html=True)
            with st.spinner("Computing word contributions…"):
                pairs = word_importance(text)
            words = text.split()
            trig = trigger_positions(words, res.get("evidence", []))
            if res["label"] == "Offensive" and 1 < len(words) <= MAX_OCCLUSIONS:
                cf, cf_p = counterfactual(text, pairs, res.get("evidence", []))
                if cf:
                    q = ", ".join(f"“{html.escape(w)}”" for w in cf)
                    st.markdown(f"<div class='box box-cf'>🔁 <b>Counterfactual</b> — without {q} the comment "
                                f"would be <b>Not offensive</b>. These words are what makes it offensive.</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<div class='box box-cf'>🔁 <b>Counterfactual</b> — no edit of up to 4 words makes it "
                                "Not offensive: the offence is carried by the sentence as a whole, not one word.</div>",
                                unsafe_allow_html=True)
            st.markdown("**Word-level contributions** (outlined = key word)")
            st.markdown("<div class='legend'><span style='background:#f6b9b9;border:1px solid #e07a7a'>toward Offensive</span>"
                        "<span style='background:#b6e2c6;border:1px solid #6cbf8b'>toward Not offensive</span>"
                        "<span style='background:#fff;outline:2px solid #12233a'>key word</span></div>",
                        unsafe_allow_html=True)
            st.markdown(render_highlight(pairs, trig), unsafe_allow_html=True)
            st.caption("Each word is removed in turn; how much the offensiveness changes is that word's "
                       "contribution (occlusion).")

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
                                "category": [p["profile"]["category"] for p in preds],
                                "target": [p["profile"]["target"] for p in preds],
                                "severity_0to6": [p["profile"]["intensity"] for p in preds]})
            st.dataframe(out, width="stretch", hide_index=True)
            st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

with tab_c:
    st.markdown("""
### About
An explainable **hybrid** system that flags **offensive language** in Sinhala, English and
code-mixed (Singlish) text. A fine-tuned **XLM-RoBERTa** model (SOLD dataset + transliteration and
contrastive augmentation, class-weighted fine-tuning) reads the language, and a transparent
**linguistic safety layer** built on Sinhala grammar adds structured knowledge of threats and identity hate
and explains each decision.

**What the safety layer recognises:**
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
- Sees through **filter evasion**: stretched letters (*kiiiill*), spacing (*u m b a*), dots, leetspeak
  (*p@ko*), zero-width characters, Cyrillic look-alike letters and vowel-dropped typing (*mrnw* = *maranawa*).
- Reads **emoji**: a weapon emoji aimed at a person (*umbata 🔪*) is a threat; an animal emoji next to a
  group noun is dehumanisation.
- Reads **meaning, not pronoun register**: Sinhala pronouns such as *meki*, *tho/thota*, *umba*, *mu* are
  casual or rude *register*, not insults. *"meki mara lassanai"* (she's really pretty) and *"thota mama kiwwa"*
  (I told you) are fine; *"meki baduwak"* or *"tho nam wandurek"* are insults because of what is SAID — the
  predicate at the end of the Sinhala clause.
- Knows **use from mention**: *"hambaya kiyanne jatiwadi wachanayak"* ("'hambaya' is a racist word")
  talks *about* a slur and is not hate speech.

**How each decision is explained:**
- an **offence profile** — the category, who is targeted (an individual, a group, or nobody in particular,
  following the OLID / SOLD annotation scheme) and a 0–6 severity level;
- the **key words** behind the decision and a **counterfactual** — the smallest set of words whose removal
  makes the comment *Not offensive*;
- **word-level contributions** by occlusion (each word is removed in turn and the change is measured).

*Final Year Research Project — BSc (Hons) Computer Science, NSBM Green University.*
""")
