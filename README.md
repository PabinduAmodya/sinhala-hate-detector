# Sinhala–English Hate Speech Detector

Explainable offensive-language detection for Sinhala, English and code-mixed (Singlish)
social-media text.

**Live demo:** https://sinhala-hate-detector.streamlit.app
**Model:** [`pabindu2003G/sinhala-hate-detector`](https://huggingface.co/pabindu2003G/sinhala-hate-detector)

## How it works
A hybrid system:

1. **Neural model.** XLM-RoBERTa-base fine-tuned in two stages with class-weighted
   cross-entropy: on SOLD plus transliterated and contrastive data (GPU), then continued on
   hand-reviewed SemiSOLD tweets and contrast pairs (CPU). Decision threshold 0.614, calibrated
   without labels and stored in the model's `config.json`.
2. **Linguistic analysis layer (`rules.py`).** Reads Sinhala morphology rather than keywords:
   violence verbs with intent suffixes, pro-drop threats, identity hate and expulsion calls,
   insults in predicate position, negation, use vs mention, counter-speech, and evasion
   normalisation (look-alike letters, spacing, leetspeak, vowel-dropped Singlish).
3. **Explanations.** Offence category, target and severity, a plain-language reason, a
   counterfactual, and an occlusion word heat-map. Occlusion was chosen after measuring it
   against human rationales (AUPRC 0.733 vs SHAP 0.534).

Borderline comments get an **"Uncertain — recommend human review"** verdict instead of a guess.

## Results (macro-F1, held-out)
| Test set | Model alone | Full system |
|---|---|---|
| SOLD test (2,000) | 0.826 | **0.833** |
| Romanized | 0.842 | 0.875 |
| Hand-labelled YouTube (500) | 0.760 | 0.798 |
| Cross-dataset SHS test (never trained on) | — | 0.768 |

Final Year Research Project — BSc (Hons) Computer Science, NSBM Green University.
