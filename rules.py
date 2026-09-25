# -*- coding: utf-8 -*-
"""
rules.py — linguistic safety layer for the Sinhala–English offensive-language detector.

WHY THIS EXISTS
  The fine-tuned XLM-RoBERTa model learned SOLD's notion of "offensive", which is
  dominated by profanity and insults. It is nearly blind to two things that matter
  most in moderation: THREATS (it scores "man umbawa maranawa" = "I'll kill you" at 6%)
  and IDENTITY-BASED HATE without a swear word ("demalu okkoma maranna ona" = "all
  Tamils must be killed" at 15%). Production moderation systems are hybrid for exactly
  this reason: a neural model for the long tail + precise, auditable rules for the
  high-harm categories.

DESIGN PRINCIPLES (lessons from auditing the previous version on SOLD)
  1. Morphology, not prefixes. The old layer matched two-letter prefixes (මර, කප, උන්),
     so මරණයට (condolences on a death), මරු (slang "awesome"), මරදාන (a place),
     කපල් (couple) and උන්වහන්සේ (honorific for a monk) all looked like violence or
     a target. Here a violence verb is ROOT + an explicit Sinhala verb SUFFIX.
  2. Intent vs report. Only intent forms count as threats: -නවා/-nawa (will), -න්නම්/-nnam
     (I will), -න්නේ/-nne (focus), -පන්/-pan (rude imperative), -මු/-mu (let's),
     -න්න ඕන/-nna ona (must). Past forms (මැරුවා/maruwa, ගැහුවා) are reports.
     Conjunctive -ලා/-la forms ("kapala") count only when chained with another violence
     verb ("kapala maranne") or with the completive auxiliary දානවා ("puchchala danawa").
  3. Sinhala is pro-drop. "කපල මරන්නේ දැන ගනින්" has NO pronoun, yet it is a death threat:
     the addressee is implied by the intimidation marker (දැනගනින් "know this",
     බලාගනින් "watch out", හිටපන් "wait") or by a rude 2nd-person imperative. The old
     layer required an explicit pronoun and missed all of these.
  4. Blockers before boosters: negation near the verb ("maranne na", "gahanna epa"),
     benign objects (chicken, hair, a phone call, a power cut), counter-speech and
     slur MENTION ("'hambaya' is a racist word") all prevent a forced verdict.
  5. Never override a confident model with a weak cue. Rescues (Offensive -> Not) only
     fire below p=0.80 and only on specific benign patterns.

Every forced verdict carries a machine-readable `rule` code that the UI turns into a
plain-language reason, so the explanation always matches the decision actually taken.
"""
import re

OFF, NOT = "Offensive", "Not offensive"

# ════════════════════════════════════════════════════════════════════════════
# 1. Normalisation & tokenisation
# ════════════════════════════════════════════════════════════════════════════
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                       "@": "a", "$": "s", "!": "i"})
_SIN = "඀-෿"


def _deobf(text):
    """Undo common filter evasion: leetspeak (p@ko, hu77a, m@ranawa) and letters split by
    punctuation (p.a.k.o). Used only for rule matching; the model sees the raw text."""
    t = str(text).lower().replace("‍", "").replace("‌", "").translate(_LEET)
    t = re.sub(r"\b(?:\w[.\-_*]+)+\w\b", lambda m: re.sub(r"[.\-_*]", "", m.group(0)), t)
    return t


def _sin_norm(w):
    """Fold spelling variants that Sinhala writers use interchangeably online:
    ණ/න and ළ/ල (කරණවා = කරනවා), and a word-final anusvara for n (බලපං = බලපන්)."""
    w = w.replace("ණ", "න").replace("ළ", "ල")
    if w.endswith("ං"):
        w = w[:-1] + "න්"
    return w


def _rom_norm(w):
    """Fold romanized-Sinhala aspiration digraphs and w/v (thota = tota, vedi = wedi).
    Doubled letters are NOT collapsed (balla 'dog' must not become bala 'look')."""
    w = re.sub(r"(.)\1{2,}", r"\1", w)          # elongation: buruwoooo -> buruwo, pakooo -> pako
    for a, b in (("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"), ("kh", "k"),
                 ("ph", "p"), ("sh", "s")):
        w = w.replace(a, b)
    return w.replace("v", "w")


def _is_sin(w):
    return any("඀" <= c <= "෿" for c in w)


_TOKEN_RX = re.compile(r"[\w" + _SIN + r"']+")
_SENT_RX = re.compile(r"[.!?\n।]+")


def tokenize(text):
    """Return (raw_lower_tokens, normalised_tokens)."""
    raw = _TOKEN_RX.findall(_deobf(text))
    norm = [_sin_norm(w) if _is_sin(w) else _rom_norm(w) for w in raw]
    return raw, norm


def sentence_ids(text):
    """Sentence index of every token (aligned with tokenize())."""
    t = _deobf(text)
    ids, sid, last = [], 0, 0
    for m in _TOKEN_RX.finditer(t):
        sid += len(_SENT_RX.findall(t[last:m.start()]))
        ids.append(sid)
        last = m.end()
    return ids


# censored obscenity: "h...nawa", "p**o", "f**k", "හු***"
_CENSORED = re.compile(r"(?:^|[\s(])(?:[hpkf][.*#_]{2,}[a-z]{1,8}|f[*#]{2,}|[හපක][.*#]{2,})")


def _joined(seq):
    """Adjacent 2- and 3-token joins, so "දැන ගනින්" = "දැනගනින්", "dana ganin" = "danaganin",
    "wanda wela yan" = "wandawelayan"."""
    return ({seq[i] + seq[i + 1] for i in range(len(seq) - 1)} |
            {seq[i] + seq[i + 1] + seq[i + 2] for i in range(len(seq) - 2)})


_W = "[\\w\u0d80-\u0dff]"      # Python's \\w misses Sinhala vowel signs (ා ු ෝ …)


def _rx(*alts):
    return re.compile(r"^(?:" + "|".join(a.replace(r"\w", _W) for a in alts) + r")$")


def script_dominant(text):
    sin = sum(1 for c in str(text) if "඀" <= c <= "෿")
    lat = sum(1 for c in str(text) if c.isascii() and c.isalpha())
    return sin > lat


# ════════════════════════════════════════════════════════════════════════════
# 2. Violence predicates (root + suffix), intent strength
# ════════════════════════════════════════════════════════════════════════════
# Romanized suffixes (on _rom_norm'd tokens). INTENT = will / I-will / focus / imperative / let's.
_R_INT = r"(?:n{1,2}awa+|n{1,2}awwa|n{1,2}a[mn]g?|n{1,2}e+|n{1,2}ema|pan+g?|piya|pal+a|pal+o|mu|yi)"
_R_INF = r"(?:n{1,2}a+)"            # infinitive: needs ona/epa/na to mean anything
_R_CNJ = r"(?:la+|l)"               # conjunctive "having X-ed"
# Script suffixes (on _sin_norm'd tokens)
_S_INT = r"(?:නවා|නව|න්නම්|න්නන්|න්නේ|න්නෙ|පන්|පිය|පල්ලා|මු|යි)"
_S_INF = r"(?:න්න)"
_S_CNJ = r"(?:ලා|ල)"

# violence roots: kill, cut, hit, burn, smash/beat, stab
_R_ROOTS = {"kill": "mara|mar(?=n)", "cut": "kapa|kap(?=n)", "hit": "gaha|gah(?=n)",
            "burn": "puc+ha|puc+a|puchcha|puca", "smash": "tala", "stab": "anina", "break": "kada"}
_S_ROOTS = {"kill": "මර", "cut": "කප", "hit": "ගහ", "burn": "පුච්ච", "smash": "තල", "stab": "ඇන",
            "break": "කඩ"}

_VPAT = {}
for _k, _r in _R_ROOTS.items():
    _VPAT[("R", _k, "int")] = _rx(f"(?:{_r}){_R_INT}")
    _VPAT[("R", _k, "inf")] = _rx(f"(?:{_r}){_R_INF}")
    _VPAT[("R", _k, "cnj")] = _rx(f"(?:{_r}){_R_CNJ}")
for _k, _r in _S_ROOTS.items():
    _VPAT[("S", _k, "int")] = _rx(f"{_r}{_S_INT}")
    _VPAT[("S", _k, "inf")] = _rx(f"{_r}{_S_INF}")
    _VPAT[("S", _k, "cnj")] = _rx(f"{_r}{_S_CNJ}")
# "මරා" / "mara" as the bare stem before the completive auxiliary (මරා දානවා / mara danawa)
_BARE_KILL = {"මරා", "mara", "maraa"}

# light-verb compounds: NOUN + kara-/tiya- ("set fire", "shoot", "finish off", "rape", "kill karanawa")
_LV_R = {"int": _rx(f"kara{_R_INT}", f"tiya{_R_INT}"),
         "inf": _rx(f"kara{_R_INF}", f"tiya{_R_INF}"),
         "cnj": _rx(f"kara{_R_CNJ}", f"tiya{_R_CNJ}")}
_LV_S = {"int": _rx(f"කර{_S_INT}", f"තිය{_S_INT}"),
         "inf": _rx("කරන්න", "තියන්න"),
         "cnj": _rx(f"කර{_S_CNJ}", f"තිය{_S_CNJ}")}
_LV_NOUNS = {  # noun -> violence kind
    "gini": "fire", "wedi": "shoot", "iwara": "finish", "ghatanaya": "kill", "gatanaya": "kill",
    "dusanaya": "rape", "dooshanaya": "rape", "rape": "rape", "kill": "kill", "murder": "kill",
    "shoot": "shoot", "stab": "stab", "attack": "hit", "nasa": "finish", "winasa": "finish", "wanasa": "finish",
    "kudu": "smash",
    "ගිනි": "fire", "වෙඩි": "shoot", "ඉවර": "finish", "ඝාතනය": "kill", "දූෂනය": "rape",
    "විනාශ": "finish", "කුඩු": "smash",
}
# bombs: noun + kara-/tiya-/gaha-/dama- in any strength
_BOMB = {"bomb", "bombs", "bombayak", "bomba", "බෝම්බ", "බෝම්බයක්", "බෝම්බය"}
_BOMB_INT = _rx(f"(?:kara|tiya|gaha|dama){_R_INT}", f"(?:කර|තිය|ගහ|දම){_S_INT}")
_BOMB_INF = _rx(f"(?:kara|tiya|gaha|dama){_R_INF}", "(?:කර|තිය|ගහ|දම)න්න")
_BOMB_CNJ = _rx(f"(?:kara|tiya|gaha|dama){_R_CNJ}", f"(?:කර|තිය|ගහ|දම){_S_CNJ}")
# body part + break/cut/smash verb ("kakul kadanawa" = break legs)
_BODY = {"kakul", "kakula", "at", "ata", "oluwa", "olua", "bella", "bela", "kata", "dat", "data",
         "කකුල්", "කකුල", "අත්", "අත", "ඔලුව", "බෙල්ල", "කට", "දත්"}
_BODY_VERB = _rx(f"(?:kada|kad(?=n)|kapa|kap(?=n)|tala|tal(?=n)){_R_INT}", f"(?:kada|kapa|tala){_R_CNJ}",
                 f"(?:කඩ|කප|තල){_S_INT}", f"(?:කඩ|කප|තල){_S_CNJ}")
# completive auxiliary "daanawa" (X-la danawa = will X completely) — intent forms only
_AUX_DANA = _rx(f"da{_R_INT}", "dana", f"දා{_S_INT}", "දාන්න")
# obligation / permission after an infinitive ("maranna ona" = must be killed)
_OBLIG = {"ona", "one", "oni", "onee", "oona", "ඕන", "ඕනෙ", "ඕනේ", "ඕනි"}

# English violence (base/future forms only — "killed", "killing" are reports/praise)
_EN_VIOL = {"kill", "murder", "stab", "shoot", "slaughter", "behead", "butcher", "rape", "burn",
            "strangle", "lynch", "bomb"}
_EN_TARGET = {"you", "u", "ya", "ur", "your", "yourself", "urself"}
_EN_NEG = {"not", "never", "wont", "won't", "dont", "don't", "wouldnt", "wouldn't", "cant", "can't"}

# ════════════════════════════════════════════════════════════════════════════
# 3. Targets, intimidation markers, negation, benign objects
# ════════════════════════════════════════════════════════════════════════════
# explicit 2nd/3rd-person human targets (nominative, accusative, dative, genitive).
# Deliberately NOT bare "oya"/"ඔය" (also "that"), "තම"/"තමා" (self), "උන්නා" (was present).
_TARGET = {
    "umba", "uba", "umbawa", "ubawa", "umbata", "ubata", "umbe", "ube", "umbage", "ubage",
    "umbala", "ubala", "umbalawa", "umbalata", "umbalage", "tho", "thou", "thopi", "thopiwa",
    "thopita", "thota", "thope", "thopila", "topi", "topiwa", "topita", "tota", "tope", "topila", "oyawa", "oyata", "oyage", "oyala", "oyalawa",
    "oyalata", "oyaa", "unwa", "unta", "unge", "uwa", "uta",
    "උඹ", "උබ", "උඹව", "උබව", "උඹට", "උබට", "උඹේ", "උබේ", "උඹගේ", "උඹල", "උඹලා",
    "උඹලව", "උඹලට", "තෝ", "තො", "තොපි", "තොපිව", "තොපිට", "තොට", "තොගේ", "තොපේ",
    "ඔයා", "ඔයාව", "ඔයාට", "ඔයාගේ", "ඔයාලා", "ඔයාලව", "ඔයාලට", "උන්ව", "උන්ට",
    "උන්ගේ", "ඌව", "උගේ", "උට",
}
# nominative/vocative only — for NAME-CALLING ("umba modaya"). Genitive "your dog" is literal.
_NOM_TARGET = {"umba", "uba", "umbala", "ubala", "tho", "thou", "thopi", "topi", "topila", "oyaa",
               "you", "u", "ya",
               "උඹ", "උබ", "උඹල", "උඹලා", "තෝ", "තො", "තොපි", "ඔයා", "ඔයාලා",
               "මේකා", "ඕකා", "අරූ", "මූ", "ඌ", "මුං", "මුන්"}
# WEAK markers double as everyday words ("balapan" = "look at this", "hitapan" = "wait")
_WEAK_MARKER = {"balapan", "balapang", "balapiya", "hitapan", "hitapang", "idapan",
                "බලපන්", "බලපිය", "හිටපන්", "ඉඳපන්"}
# intimidation / warning markers that imply an addressee (checked on tokens and joined pairs)
_MARKER = {
    "danaganin", "danaganing", "danagania", "balaganin", "balaganing", "balapan", "balapang",
    "balapiya", "hitapan", "hitapang", "matakatiyaganin", "matakatiyaganing", "lestiweyan",
    "lestiwenin", "berenaba", "berennaba", "berennebe", "hoyagena", "hoyagenawit",
    "gedarataawit", "gedarataenawa", "eliyatawaren", "eliyatawareng", "amuamuwe", "idapan",
    "දැනගනින්", "බලාගනින්", "බලපන්", "බලපිය", "හිටපන්", "මතකතියාගනින්", "ලෑස්තිවෙයන්",
    "බේරෙන්නබෑ", "හොයාගෙන", "ගෙදරටඇවිත්", "ගෙදරටඑනවා", "එලියටවරෙන්", "අමුඅමුවේ", "ඉඳපන්",
}
_RUDE_IMP = _rx(r"[a-z]{2,}apan+g?", r"[a-z]{2,}apiya", r"[a-z]{2,}apal+[ao]", r"[a-z]*ganin+g?",
                r"waren+g?", r"palayan", r"yapan",
                r"[" + _SIN + r"]+(?:පන්|පල්ලා|ගනින්)", r"[" + _SIN + r"]{2,}පිය", "වරෙන්", "පලයන්")
_RUDE_NOT = {"japan", "ජපන්", "දෙමාපිය", "මාපිය", "පිය"}
_NEG_AFTER = {"na", "naa", "nae", "naha", "nehe", "neha", "epa", "ba", "bae", "ne",
              "නෑ", "නැහැ", "නැ", "නැත", "එපා", "බෑ", "බැහැ"}
# benign objects/contexts that make a violence verb literal (food, pests, hair, trees,
# services, sport, games, time). Normalised forms.
_BENIGN = {
    "kukula", "kukulawa", "kukulo", "kukulan", "malu", "maluwa", "eluwa", "maduruwo",
    "maduruwa", "maduru", "kumbiyo", "kakuluwo", "cake", "kek", "pan", "kema", "kari",
    "curry", "elawalu", "pol", "konde", "niya", "rawula", "gas", "gasa", "gaha", "kola",
    "current", "karant", "light", "watura", "line", "call", "kol", "phone", "ticket", "tiket",
    "bill", "bola", "ball", "bole", "selfie", "photo", "horn", "bell", "time", "game", "pubg",
    "freefire", "match", "level", "spray", "beet", "boot", "wicket", "wickets", "run", "runs",
    "six", "sixes", "four", "fours", "cricket", "bat", "batting", "innings", "over", "overs",
    "arakku", "beer", "bear", "ganja", "kansa", "joint", "smoke", "shot", "shots", "peg", "pegs",
    "potoo", "poto", "fb", "status", "post", "paara", "hita", "hitata", "hite", "liyum",
    "letter", "letters", "shooting", "aluwa", "biriyani", "buriyani", "data", "series",
    "partnership", "roti", "rotti", "bat", "bulat",
    "කුකුලා", "කුකුලාව", "කුකුලන්", "මාලු", "එලුවා", "මදුරුවෝ", "මදුරුවා", "කූඹියෝ",
    "කකුලුවෝ", "කේක්", "පාන්", "කරියක්", "එලවලු", "පොල්", "කොන්ඩය", "නිය", "රැවුල",
    "ගස්", "ගහ", "කොල", "කරන්ට්", "වතුර", "ලයිට්", "කෝල්", "ටිකට්", "බිල්", "බෝලෙට",
    "බෝලය", "බෝල", "සෙල්ෆි", "ෆොටෝ", "පොටෝ", "ටයිම්", "ගේම්", "බෙහෙත්", "ක්රිකට්", "විකට්",
    "ලකුනු", "සික්සර්", "මැච්", "මැච්එක", "අරක්කු", "බියර්", "කංසා", "ගංජා", "ස්ටේටස්",
    "පාර", "පාරක්", "හිත", "හිත්", "හිතට", "හදවත", "ලියුම්", "ලියුමක්", "කපල්",
    "අලුවා", "බුරියානි", "කැලේ", "කැලෑව", "ඩේටා", "සීරීස්", "කේක්එක", "රොටී", "රොටි", "බත්",
    "බුලත්", "බුලත්විටක්",
}

# ════════════════════════════════════════════════════════════════════════════
# 4. Curses / death wishes (offensive whether or not a target is named)
# ════════════════════════════════════════════════════════════════════════════
_CURSE_TOK = {"mariyan", "mareyan", "mariyang", "maren", "mareng", "kys",
              "මැරියන්", "මැරෙයන්", "මැරෙන්"}
_CURSE_PAIR = {  # joined adjacent tokens
    "marilayan", "marilapalayan", "malapalayan", "mailapalayan", "wandawelayan", "wandawenna",
    "apayatayan", "henagahapan", "henagahapiya",
    "henamagahapiya", "henamagahapan", "godie", "youdie", "dropdead",
    "මැරිලායන්", "මැරිලාපලයන්", "වඳවෙලායන්", "අපායටයන්", "හෙනගහපන්",
    "හෙනගහපිය", "හෙනමගහපිය", "හෙනමගහපන්",
}
_EN_THREAT_PHRASES = [("deserve", "to", "be", "raped"), ("deserves", "to", "be", "raped"),
                      ("should", "be", "raped"), ("should", "get", "raped"), ("get", "raped"),
                      ("you're", "dead"), ("youre", "dead"), ("you", "are", "dead"), ("ur", "dead"),
                      ("u", "r", "dead"), ("watch", "your", "back"), ("watch", "ur", "back"),
                      ("know", "where", "you", "live"), ("gonna", "hurt", "you"), ("will", "hurt", "you"),
                      ("beat", "you", "up"), ("break", "your", "legs")]
# bare "hena gahapiya/gahapan" is also an exclamation ("damn it!") — needs an addressee (see analyse)
_HENA_PAIRS = {"henagahapan", "henagahapiya", "henamagahapiya", "henamagahapan",
               "හෙනගහපන්", "හෙනගහපිය", "හෙනමගහපිය", "හෙනමගහපන්"}
_CURSE_EN = [("hope", "you", "die"), ("kill", "yourself"), ("kill", "urself")]

# ════════════════════════════════════════════════════════════════════════════
# 5. Identity groups, slurs and hostile predicates
# ════════════════════════════════════════════════════════════════════════════
# SLURS — derogatory in themselves (History Workshop 2020; CPA / Hashtag Generation reports):
# hambaya/hambankarayo, thambiya/thambila (NOT bare "thambi" = "younger brother" in Tamil),
# marakkalaya, kallathoni ("illegal boat-person"), "para demala" ("alien Tamil").
_SLUR = _rx(r"hambay[aoe]\w*", r"hambankar\w*", r"tambil\w*", r"tambiy[aoe]\w*", r"marakkal\w*",
            r"kal+aton\w*", r"kal+atoni\w*",
            r"හම්බය\w*", r"හම්බයෝ", r"හම්බන්කාර\w*", r"තම්බිල\w*", r"තම්බියා\w*", r"තම්බියෝ",
            r"මරක්කල\w*", r"කල්ලතෝනි\w*",
            # caste (Rodiya = lit. "filth"; historically untouchable) and LGBTQ slurs
            r"rodiy[aoe]\w*", r"rodiyek\w*", r"berawaya\w*", r"berawayo\w*", r"berawayek\w*",
            r"nachchi\w*", r"nachi\w*", r"nacci\w*",
            r"රොඩියා\w*", r"රොඩියෙක්\w*", r"රොඩියෝ\w*", r"බෙරවායා\w*", r"බෙරවායෝ\w*", r"බෙරවායෙක්\w*",
            r"නච්චි\w*")
# neutral identity nouns — hateful only with a hostile predicate
_GROUP = _rx(r"demal\w*", r"tamil\w*", r"muslim\w*", r"sinhalay\w*", r"sinhalun\w*", r"kitunu\w*",
             r"christian\w*", r"hindu\w*", r"moor\w*", r"sinhalese",
             r"දෙමල\w*", r"මුස්ලිම්\w*", r"සිංහලයෝ", r"සිංහලයින්\w*", r"කිතුනු\w*", r"හින්දු\w*")
_PLACE = _rx(r"pal+i\w*", r"kowil\w*", r"pansal\w*", r"dewal\w*",
             r"පල්ලි\w*", r"කෝවිල්\w*", r"පන්සල්\w*", r"දේවාල\w*")
_PARA = {"para", "පර"}
_KOTI = {"koti", "kotiyo", "කොටි", "කොටියෝ"}            # "Tiger(s)" — slur only next to a group noun
# hostile predicates about a group
_EXPEL_TOK = _rx(r"elawan+a", r"palawan+a", r"yawan+a", r"pan+an+a", r"palayal+a", r"palayal+o",
                 r"elawapan", r"එලවන්න", r"පලවන්න", r"යවන්න", r"පන්නන්න", r"පලයල්ලා")
# "from/out of the country" phrases that turn "go" into an expulsion ("me raten palayan")
_FROM_COUNTRY = _rx(r"ratin", r"raten", r"ratayen", r"ratenma", r"uturata", r"indiyawata", r"arabiyata",
                    r"රටින්", r"රටෙන්", r"උතුරට", r"ඉන්දියාවට", r"අරාබියට")
_EXPEL_PAIR = {"idakna", "idaknae", "idana", "idanae", "ratadenaepa", "ratadennaepa", "ratadennaba",
               "ඉඩක්නෑ", "ඉඩනෑ", "රටදෙන්නඑපා"}
_DEHUM = _rx(r"bal{2}o+", r"bal{2}an", r"uro+", r"kunu", r"parasites?", r"animals", r"vermin",
             r"cockroach\w*", r"pigs", r"dogs", r"horu", r"kapatiyo", r"saturo", r"saturu",
             r"jihad\w*",
             r"බල්ලෝ", r"බල්ලො", r"බල්ලන්", r"ඌරෝ", r"ඌරො", r"කුනු", r"හොරු", r"කපටියෝ",
             r"සතුරෝ", r"ජිහාද්\w*")
_DEHUM_PAIR = {"bowenawa", "bowena", "rataallanawa", "rataalla", "wandapeti", "wandakarana",
               "wandakaranna", "බෝවෙනවා", "රටඅල්ලනවා", "වඳපෙති", "වඳකරන්න"}
_BOYCOTT_SHOP = _rx(r"kade\w*", r"kadaw\w*", r"කඩවල\w*", r"කඩේ\w*")
_BOYCOTT_ACT = {"yannaepa", "yanaepa", "gannaepa", "ganaepa", "kannaepa", "kanaepa",
                "යන්නඑපා", "ගන්නඑපා", "කන්නඑපා"}
_EN_GROUP = {"muslim", "muslims", "tamil", "tamils", "sinhalese", "christians", "hindus", "moors"}
_EN_HOSTILE = {"kill", "killed", "murder", "murdered", "parasites", "animals", "vermin", "pigs",
               "dogs", "cockroaches", "terrorists", "exterminate", "exterminated", "deport",
               "deported", "wipe", "wiped"}
# counter-speech / mention markers — a group + violence sentence that OPPOSES it
_COUNTER = _rx(r"waradi\w*", r"waradak", r"waradda\w*", r"nawatamu", r"sahodara\w*",
               r"jatiwad\w*", r"wrong", r"stop", r"against", r"peace", r"respect",
               r"වැරදි\w*", r"වැරැද්ද\w*", r"සහෝදර\w*", r"නවත්තමු", r"ජාතිවාද\w*")
_MENTION = _rx(r"kiyana", r"kiyane", r"kiyanne", r"wacanay\w*", r"word",
               r"කියන්නේ", r"කියන", r"වචනය\w*")

# ════════════════════════════════════════════════════════════════════════════
# 6. Personal abuse lexicons (inflection-aware)
# ════════════════════════════════════════════════════════════════════════════
# Unambiguous obscenities — offensive in any context. Script stems carry an explicit
# suffix set so පකෝ / පකයා / පකයෝ match but පකිස්තානය (Pakistan) does not.
_VULGAR = _rx(
    r"pak+o+", r"paka", r"pakaya", r"pakayo", r"pakay[ae]", r"puka", r"puke", r"pukmanta",
    r"keri", r"keriya", r"hut+a", r"hut+o+", r"hut+ige", r"hut+ek", r"wesi", r"wesiya",
    r"wesige", r"wesawa", r"ponnaya", r"ponnayo", r"ponnaye", r"kariya", r"kimba", r"labba",
    r"kukku", r"hukan+a", r"hukanawa", r"hukapan+g?", r"hukala", r"huka", r"tauka",
    r"taukanawa", r"konakapala", r"junda", r"ambakissa",
    r"ht+o+", r"ht+a", r"pko+", r"pky[ao]?", r"pkaya", r"ponnya", r"hkapan+g?",
    r"ponnay\w*", r"pakay\w*", r"pake", r"pakek\w*", r"pakata", r"pakage", r"kariye?k\w*",
    r"kariyo", r"kariyage", r"hut+ek\w*", r"wesiy\w*", r"pakaden+\w*",
    r"wal+apat+a", r"walat+aya", r"lowanawa",
    r"fuck\w*", r"bitch\w*", r"asshole", r"slut", r"whore", r"cunt",
    r"පක(?:ා|ෝ|ො|යා|යෝ|යො|යින්|යන්ට|යට|යෙක්|යෙක්ද|ෙක්|ෙක්ද)?", r"පොන්නය(?:ා|ෝ|ො|ින්|ට|ෙක්|ෙක්ද)?",
    r"පූක(?:ා|ේ|ට)?", r"කැරියෙක්\w*", r"වේසියෙක්\w*",
    r"හුත්ත(?:ා|ෝ|ො|ිගේ|ෙක්)?", r"වේසි(?:යා|ගේ|යෝ)?", r"වේසාවා", r"කැරි(?:යා|යෝ)?",
    r"පුක(?:ේ|ට)?", r"ලබ්බ(?:ා|ෙ)?", r"හුකන(?:වා|්න)", r"හුකපන්", r"හුකලා",
    r"කිම්බ(?:ා)?", r"පුක්මන්ත(?:ා)?", r"උක්නවා", r"උකනවා", r"උක්කනවා", r"හ්කන\w*",
)
# noun-form name-calls — offensive when aimed at a person (nominative target)
_HARD = _rx(
    r"modaya", r"modayo", r"modayek", r"modayaa", r"gonaa", r"gonek", r"buruwa", r"booruwa",
    r"buruwek", r"bal{2}a", r"bal{2}o", r"bal{2}ek", r"pis{2}a", r"pis{2}ek", r"pis{2}i", r"pis{2}u", r"harakaa",
    r"gawaya", r"idiot", r"bastard", r"moron", r"retard", r"loser",
    r"buruwo+", r"modayo+", r"gon+u", r"harako+", r"kalakan+i\w*", r"yak+u", r"karumay\w*",
    r"karumak+aray\w*", r"sak+iliy[ao]", r"sapayak", r"shapayak", r"hypocrite", r"liar", r"clown", r"traitor",
    r"මෝඩයා", r"මෝඩයෙක්", r"බූරුවා", r"බූරුවෙක්", r"ගොනා", r"ගොනෙක්", r"බල්ලා", r"බල්ලො",
    r"බල්ලෙක්", r"පිස්සා", r"පිස්සෙක්", r"හරකා", r"හරකෙක්", r"බූරුවෝ", r"මෝඩයෝ", r"ගොන්නු",
    r"කාලකන්නි\w*", r"යක්කු", r"කරුමය\w*", r"කරුමක්කාර\w*", r"සක්කිලියා", r"සක්කිලියෝ",
    r"ශාපයක්", r"සාපයක්", r"ශාපය", r"සාපය",
)
# person nouns — a SOFT adjective right next to one is an insult ("gon gani" = stupid woman)
_PERSON = {"gani", "gaani", "ganu", "ganiyak", "ganiyek", "gaaniyak", "ganiyo", "miniha", "minihek", "minisu", "minissu", "ekek",
           "ගෑනියක්", "ගෑනියෙක්", "ගෑනියෝ",
           "kolla", "kella", "kollo", "kello", "lamaya", "unge",
           "ගෑනි", "ගෑනු", "මිනිහා", "මිනිහෙක්", "මිනිස්සු", "එකා", "එකෙක්", "කොල්ලා", "කෙල්ල"}
# SOFT adjectives — insult only when aimed at a person
_SOFT = _rx(r"moda", r"gon", r"gona", r"buru", r"modai", r"gonai", r"මෝඩ", r"ගොන්", r"බුරු")
_ANIMAL = _rx(r"haraka", r"ura", r"uura", r"wandura+", r"balu", r"හරක", r"හරකා", r"ඌරා", r"වඳුරා", r"බලු")
_ENDEAR = {"mage", "ape", "adare", "adarei", "cooti", "chooti", "rattaran", "punci", "punchi",
           "sudu", "මගේ", "ආදරේ", "පුංචි", "චූටි"}
_CHILD = {"patiya", "patiyo", "kolla", "kella", "duwa", "puta", "baba", "malli", "nangi",
          "පැටියා", "කොල්ලා", "දුව", "පුතා", "බබා"}
_FRIENDLY = {"macan", "machan", "macang", "macho", "bosa", "boss", "bro", "brother", "yaluwa",
             "yaaluwa", "aiya", "malli", "nangi", "patiya", "patiyo", "cooti", "puta", "ela",
             "supiri", "niyamai", "superb", "adarei", "raja", "elakiri", "homies", "homie", "bruh", "dude", "bros", "broo",
             "මචන්", "බොස", "අයිය", "මල්ලි", "පැටියා", "එලකිරි"}
# hostile imperatives ("get lost", "shut up") — never rescue a message that contains one
_HOSTILE_IMP = {"palayan", "palayang", "palayal", "palayalla", "palayallo", "yapan", "wahapan",
                "wahapang", "wahapiya", "wahapiyaw", "පලයන්", "පලයල්ලා", "යපන්", "වහපන්", "වහපිය"}
# THING nouns: a mild word describing one of these is not a personal insult ("moda wada")
_THING = {"wada", "weda", "wade", "wede", "wadak", "wedak", "katawa", "kata", "deyak", "dewal",
          "prasna", "prasnaya", "question", "idea", "plan", "video", "joke", "comment", "post",
          "wadda", "ewa", "tiranaya", "kiyana",
          "වැඩ", "වැඩක්", "වැඩේ", "කතාව", "කතා", "දෙයක්", "දේවල්", "ප්රශ්න", "වීඩියෝ", "තීරනය"}
# fillers allowed between a dative target and "gaha-" ("umbata ekak gahannam" = I'll give you one)
_HIT_FILL = {"hodata", "hondata", "sirawatama", "tawa", "ekak", "dekak", "hari", "aye", "ayet",
             "dan", "denma", "honda", "hoda", "ahuwena", "aniwa", "හොඳට", "තව", "එකක්", "දෙකක්",
             "ආයෙත්", "දැන්", "හරි"}
# possessive "your" — "oyage konde kapannam" = I'll cut YOUR HAIR (the object is a thing)
_GENITIVE = {"umbe", "ube", "umbage", "ubage", "oyage", "thoge", "toge", "thope", "tope", "unge",
             "oyalage", "umbalage", "උඹේ", "උබේ", "උඹගේ", "ඔයාගේ", "තොගේ", "තොපේ", "උන්ගේ",
             "ඔයාලගේ", "උඹලගේ"}
# figurative "dying": dying of laughter / hunger / tiredness / embarrassment / longing
_DYING = {"marenawa", "marenawaa", "marenna", "marenne", "marila", "මැරෙනවා", "මැරෙනව", "මැරෙන්න",
          "මැරෙන්නේ", "මැරිලා"}
_DYING_CTX = {"hina", "hinawela", "badaginne", "badagini", "mahansiyen", "mahansi", "lajjawe", "lajjawen",
              "asawe", "asawen", "wada", "karala", "හිනා", "බඩගින්නේ", "බඩගිනි", "මහන්සියෙන්", "මහන්සි",
              "ලැජ්ජාවේ", "ලැජ්ජාවෙන්", "ආසාවේ", "ආසාවෙන්", "වැඩ", "කරලා"}
# religious / cultural observances — both models over-flag them (identity-term bias, Dixon et al. 2018):
# "vesak day" 0.98, "Eid mubarak" 0.60. A mention with NO abusive cue at all is cleared.
_FESTIVAL = {"vesak", "wesak", "poson", "poya", "esala", "perahera", "kathina", "dansal", "dansala", "christmas",
             "xmas", "easter", "ramazan", "ramadan", "ramzan", "eid", "mubarak", "iftar", "hajj", "deepavali",
             "diwali", "pongal", "thaipongal", "navarathri", "shivarathri", "awurudu", "awurudda", "avurudu",
             "avurudda", "nallur", "kataragama", "madhu",
             "වෙසක්", "පොසොන්", "පෝය", "ඇසල", "පෙරහැර", "කඨින", "දන්සල්", "නත්තල්", "පාස්කු", "රාමසාන්",
             "ඊද්", "ඉෆ්තාර්", "දීපවාලි", "තෛපොංගල්", "පොංගල්", "අවුරුදු", "අවුරුද්ද", "අවුරුද්දට"}
# supportive verbs aimed at "you": save / help / look after / love
_SUPPORT = {"beraganna", "beragannam", "beragannawa", "udaw", "udawu", "balagannam", "balaganna",
            "adarei", "adare", "raka", "rakinawa", "බේරගන්නම්", "බේරගන්නවා", "බේරගන්න", "උදව්",
            "බලාගන්නම්", "ආදරෙයි", "රකිනවා", "රැකගන්නම්"}
_CASUAL = {"umba", "uba", "umbala", "umbata", "tho", "thopi", "thou", "macan", "machan", "macho",
           "malli", "aiya", "yaluwa", "yaaluwa", "bro", "ban", "bn"}


def _match(rx, toks):
    for w in toks:
        if rx.match(w):
            return w
    return None


# ════════════════════════════════════════════════════════════════════════════
# 7. Analysis
# ════════════════════════════════════════════════════════════════════════════
def _violence_preds(raw, seq):
    """Find violence predicates as (noun_or_verb_index, kind, strength, verb_index).
    strength: int (intent), obl (infinitive + must), inf (bare infinitive), cnj (conjunctive
    "having X-ed"), en (English base-form verb)."""
    preds = []
    for i, w in enumerate(seq):
        script = "S" if _is_sin(w) else "R"
        for kind in ("kill", "cut", "hit", "burn", "smash", "stab", "break"):
            for st in ("int", "inf", "cnj"):
                if _VPAT[(script, kind, st)].match(w):
                    preds.append((i, kind, st, i))
        if w in _BARE_KILL and i + 1 < len(seq) and _AUX_DANA.match(seq[i + 1]):
            preds.append((i, "kill", "int", i + 1))
        # light verbs: NOUN + kara-/tiya-  ("gini tiyanawa", "wedi tiyanawa", "kill karanawa")
        lv = _LV_NOUNS.get(w) or _LV_NOUNS.get(raw[i])
        if lv and i + 1 < len(seq):
            nxt = seq[i + 1]
            lvs = _LV_S if _is_sin(nxt) else _LV_R
            for st in ("int", "inf", "cnj"):
                if lvs[st].match(nxt):
                    preds.append((i, lv, st, i + 1))
                    break
        # bombing: "bomb karanna / gahanawa / tiyanawa", "බෝම්බ ගහනවා / තියනවා" (2019 Easter attacks)
        if (w in _BOMB or raw[i] in _BOMB) and i + 1 < len(seq):
            j = i + 2 if seq[i + 1] in {"ekak", "eka", "ekk", "එකක්", "එක"} and i + 2 < len(seq) else i + 1
            for st, rx in (("int", _BOMB_INT), ("inf", _BOMB_INF), ("cnj", _BOMB_CNJ)):
                if rx.match(seq[j]):
                    preds.append((i, "fire", st, j))
                    break
        # body part + break/cut verb ("kakul kadanawa" = break (your) legs)
        if w in _BODY and i + 1 < len(seq) and _BODY_VERB.match(seq[i + 1]):
            cnj_form = bool(re.search(r"(?:la+|l|ලා|ල)$", seq[i + 1]))
            preds.append((i, "body", "cnj" if cnj_form else "int", i + 1))
        if raw[i] in _EN_VIOL:
            preds.append((i, "en", "en", i))
    out = []
    for (i, k, st, vi) in preds:
        nxt = seq[vi + 1] if vi + 1 < len(seq) else ""
        nxt2 = seq[vi + 2] if vi + 2 < len(seq) else ""
        if st == "inf" and (nxt in _OBLIG or (nxt in {"danna", "dana", "දාන්න"} and nxt2 in _OBLIG)):
            st = "obl"                                   # maranna ona = must be killed
        if st == "cnj" and nxt and _AUX_DANA.match(nxt):
            st = "int"                                   # puchchala danawa = will burn (you) up
        if st == "cnj" and nxt in {"danna", "දාන්න"} and nxt2 in _OBLIG:
            st = "obl"                                   # marala danna ona = must be killed off
        out.append((i, k, st, vi))
    return out


def _negated(i, vi, raw, seq):
    """Is the predicate negated? Sinhala negates right AFTER the verb (maranne na, gahanna epa)
    or after its auxiliary (maranna ona na, marala danna epa); English before it (won't kill).
    A wider window would wrongly treat "maranawa, berenna ba" ("...you won't escape") as negated."""
    nxt = seq[vi + 1] if vi + 1 < len(seq) else ""
    nxt2 = seq[vi + 2] if vi + 2 < len(seq) else ""
    if nxt in _NEG_AFTER:
        return True
    if (nxt in _OBLIG or nxt in {"danna", "dana", "දාන්න"}) and nxt2 in _NEG_AFTER:
        return True
    before = raw[max(0, i - 3):i]
    return any(w in _EN_NEG for w in before)


_GRAVE = {"kill", "burn", "rape", "fire", "shoot", "stab"}
# "you" is the SUBJECT, not the victim, when a 1st-person object is present ("umba mawa kapala dannne")
_FIRST_OBJ = {"mawa", "mawat", "mata", "apiwa", "apiwat", "apita", "මාව", "මාවත්", "මට", "අපිව", "අපිට"}
_QUESTION = {"da", "de", "ද", "දෙ", "ඩ"}
_HEDGE = {"wage", "wge", "wagee", "වගේ", "වගෙ", "වාගේ"}
_REPORT = {"kiyala", "kiyla", "kiwwa", "kiuwa", "kiyai", "කියයි", "අමතයි", "කියා", "කියලා", "කියල",
           "කිව්වා", "කිව්වේ", "කිවුවා", "කිවුවේ", "තර්ජනය", "tarjanaya"}
# narration markers valid ANYWHERE in the sentence (news reports, stories, quoted threats)
_NARRATION = {"යැයි", "කියමින්", "තර්ජනය", "තර්ජන", "කළේය", "කීවේය", "පැවසීය", "දුන්නේය", "කිව්වලු",
              "කිව්වාලු", "අමතයි", "කියයි", "tarjanaya", "threatened", "threat"}
# idioms: "diya redden bella kapanawa" = quiet betrayal; "kata kapala" = open-mouthed / palate
_IDIOM_PREV = {"රෙද්දෙන්", "redden", "reddeen"}
_CALL_KINDS = {"kill", "burn", "fire", "shoot", "rape"}
_FUTURE_1SG = re.compile(r"(?:n{1,2}a[mn]g?|න්නම්|න්නන්)$")


def analyse(text):
    """Compute every linguistic signal once. The verdict AND the explanation both read this,
    so the stated reason can never disagree with the decision."""
    raw, seq = tokenize(text)
    n = len(seq)
    sid = sentence_ids(text)
    tokset = set(seq) | set(raw)
    pairs = _joined(seq) | _joined(raw)
    s = {"raw": raw, "seq": seq}

    tgt_idx = [i for i in range(n) if seq[i] in _TARGET or raw[i] in _TARGET or raw[i] in _EN_TARGET]
    target = [raw[i] for i in tgt_idx]
    nom_idx = [i for i in range(n) if seq[i] in _NOM_TARGET or raw[i] in _NOM_TARGET]
    # dative/accusative forms (umbata, thota, umbawa, උඹට, තොපිව) — the OBJECT of the action
    obj_idx = [i for i in tgt_idx if seq[i].endswith(("ta", "wa", "ට", "ව")) or raw[i].endswith(("ta", "wa"))]
    marker = sorted((tokset | pairs) & _MARKER)
    strong_marker = [m for m in marker if m not in _WEAK_MARKER]
    # sentences that contain a strong marker (single token or a 2-3 token join)
    marker_sids = set()
    for L in (1, 2, 3):
        for i in range(n - L + 1):
            j = "".join(seq[i:i + L])
            if j in _MARKER and j not in _WEAK_MARKER:
                marker_sids.add(sid[i])
    rude = [i for i in range(n) if _RUDE_IMP.match(seq[i]) and seq[i] not in _RUDE_NOT
            and not seq[i].endswith("මාපිය")]
    benign = sorted(tokset & _BENIGN)

    preds = _violence_preds(raw, seq)
    live = [p for p in preds if not _negated(p[0], p[3], raw, seq)]
    negated_any = len(live) < len(preds)
    person_live = [p for p in live if p[1] != "break"]      # "break" = against places, not people
    strong = [p for p in person_live if p[2] in ("int", "obl")]
    en_viol = [p for p in person_live if p[2] == "en"]

    def blocked(p):
        """Question ("maranawada?"), hedge ("marai wage" = looks like he'd kill), or reported speech
        ("... maranawa kiyala / – X අමතයි") — none of these is a direct threat."""
        i, k, st, vi = p
        v = seq[vi]
        nxt = seq[vi + 1:vi + 4]
        same = [w for j, w in enumerate(seq) if sid[j] == sid[vi]]
        if v.endswith(("da", "ද")) and len(v) > 4 or (nxt and nxt[0] in _QUESTION):
            return True
        # question clitic on the next word ("tiyanne adada?", "maranawa hetada")
        if nxt and len(nxt[0]) > 2 and nxt[0].endswith(("ද", "da")) and nxt[0] not in {"ada", "ida"}:
            return True
        after5 = seq[vi + 1:vi + 6]
        if set(nxt[:2]) & _HEDGE or set(after5) & _REPORT or set(same) & _NARRATION:
            return True
        if any(w in _IDIOM_PREV for w in seq[max(0, i - 3):i]):
            return True
        return "?" in text and bool(set(same) & _QUESTION)

    def targeted(p):
        """An explicit target sits right before the verb (Sinhala is SOV) or, colloquially, just
        after it — and always in the SAME sentence. For the polysemous 'gaha-' (hit / call /
        smoke / bat / stamp) the dative/accusative object must sit DIRECTLY before the verb
        ("thota gahanawa", "umbata ekak gahannam"); "umbata call ekak gahannam" has a thing
        (call) in that slot, so it is not violence."""
        i, k, _, vi = p
        if k in ("hit", "cut"):
            # polysemous verbs (hit/call/bat/smoke; cut/cut-off/cut-sweets): the VICTIM must be a
            # dative/accusative object right before the verb ("thota gahanawa", "umbawa kapala")
            j = vi - 1
            if j >= 0 and seq[j] in _HIT_FILL:
                j -= 1
            return j >= 0 and j in obj_idx and sid[j] == sid[vi]
        lo = min(i, vi) - 4
        near = [t for t in tgt_idx if lo <= t <= vi + 2 and t not in (i, vi) and sid[t] == sid[vi]]
        if not near:
            return False
        if any(t in obj_idx for t in near):
            return True
        # only a nominative "umba/tho" nearby: colloquially the object ("tho maranawa" = I'll kill you)
        # unless a 1st-person object shows "you" is the subject ("tho mawa marapiya", "tho apiwa maranawa")
        same = {seq[j] for j in range(n) if sid[j] == sid[vi]}
        return not (same & _FIRST_OBJ)

    # ── threat against a person ──
    threat = None
    strong = [p for p in strong if not blocked(p)]
    if strong or en_viol:
        tgt_preds = [p for p in strong if targeted(p)]
        serial = any(p[2] == "cnj" and any(q[3] in (p[3] + 1, p[3] + 2) and sid[q[3]] == sid[p[3]]
                                          for q in strong)
                     for p in person_live)                   # gahala maranawa, kapala maranne
        body = any(p[1] == "body" for p in strong)
        first_future = any(p[2] == "int" and _FUTURE_1SG.search(seq[p[3]]) and p[1] in _GRAVE | {"smash", "finish"}
                           for p in strong)                  # marannam = I WILL kill
        en_threat = any(any(w in _EN_TARGET for w in raw[p[3] + 1:p[3] + 4]) for p in en_viol)
        own = {p[3] for p in strong}
        other_rude = [i for i in rude if i not in own]
        grave_any = any(p[1] in _GRAVE for p in strong)
        acc_person = any(p[1] in _GRAVE and targeted(p) for p in strong) and \
            any(t.endswith(("wa", "ව")) for t in target)
        literal = bool(benign) and not acc_person
        if not literal:
            if tgt_preds:
                threat = "threat_target"
            elif en_threat:
                threat = "threat_target"
            elif serial or body or first_future:
                threat = "threat_implied"
            elif strong and strong_marker and any(sid[p[3]] in marker_sids for p in strong):
                threat = "threat_implied"
            elif any(p[2] == "obl" and p[1] in _CALL_KINDS for p in strong):
                threat = "violence_call"
    if threat is None and not benign:
        for pat in _EN_THREAT_PHRASES:
            m = len(pat)
            if any(tuple(raw[i:i + m]) == pat for i in range(len(raw) - m + 1)) and \
                    not any(w in _EN_NEG for w in raw):
                threat = "threat_target"
    s["threat"] = threat

    # ── curse / death wish ──
    reported = {"kiyala", "kiyla", "kiyalaa", "kiwwa", "kiuwa", "කියල", "කියලා", "කිව්වා", "කිව්ව"}
    person_dat = {"ammata", "appata", "taattata", "ammata", "aiyata", "akkata", "maluta", "unta", "muta",
                  "uta", "eyata", "eyaata", "minihata", "ganita", "kellata", "kollata", "ekita", "aantita",
                  "අම්මට", "තාත්තට", "අයියට", "අක්කට", "මූට", "ඌට", "උට", "එයාට", "මිනිහට", "ගෑනිට",
                  "කෙල්ලට", "කොල්ලට", "ඒකිට", "ඇන්ට්ට", "ඇන්ටිට"}

    def addressee_dative(w):
        return w in _TARGET and w.endswith(("ta", "ට")) or w in person_dat or \
            (w.endswith(("tama", "ටම")) and (w[:-2] in person_dat or w[:-2] in _TARGET or
                                             w.startswith(("okkota", "ඔක්කොට"))))

    def hena_curse(i):
        if not (seq[i] in {"hena", "henama", "හෙන", "හෙනම"} and i + 1 < n and
                (seq[i + 1].startswith("gaha") or seq[i + 1].startswith("ගහ"))):
            return False
        after = set(seq[i + 2:i + 4])
        if after & reported or after & {"wage", "wge", "වගේ", "වගෙ"}:
            return False                                     # reported speech / simile ("like lightning")
        return (i > 0 and addressee_dative(seq[i - 1])) or any(sid[t] == sid[i] for t in tgt_idx)

    hena_addressed = any(hena_curse(i) for i in range(n))
    tok_curse = bool(tokset & _CURSE_TOK)
    pair_curse = bool(pairs & (_CURSE_PAIR - _HENA_PAIRS))
    curse = tok_curse or pair_curse or hena_addressed
    for pat in _CURSE_EN:
        m = len(pat)
        if any(tuple(raw[i:i + m]) == pat for i in range(len(raw) - m + 1)):
            curse = True
    s["curse"] = curse

    # ── identity-based hate ──
    slur = _match(_SLUR, seq)
    group = _match(_GROUP, seq) or ([w for w in raw if w in _EN_GROUP] or [None])[0] or slur
    place = _match(_PLACE, seq)
    para_group = any(seq[i] in _PARA and _GROUP.match(seq[i + 1]) for i in range(n - 1))
    koti_group = any(seq[i] in _KOTI and _GROUP.match(seq[i + 1]) for i in range(n - 1))
    counter = bool(_match(_COUNTER, seq)) or negated_any
    mention = bool(_match(_MENTION, seq))
    # the group noun must be CLOSE to its hostile predicate (same sentence, <= 5 tokens):
    # long news/conspiracy posts mention a group in one place and a killing in another
    gidx = [i for i in range(n) if _SLUR.match(seq[i]) or _GROUP.match(seq[i]) or _PLACE.match(seq[i])
            or raw[i] in _EN_GROUP]

    def near_group(j, d=5):
        return any(abs(j - g) <= d and sid[j] == sid[g] for g in gidx)

    expel = any((_EXPEL_TOK.match(seq[j]) or
                 (seq[j] in {"palayan", "yanna", "පලයන්", "යන්න"} and
                  any(_FROM_COUNTRY.match(seq[k]) for k in range(max(0, j - 3), j))))
                and near_group(j) for j in range(n)) or bool(pairs & _EXPEL_PAIR)
    dehum = any(_DEHUM.match(seq[j]) and near_group(j, 3) for j in range(n)) or bool(pairs & _DEHUM_PAIR)
    boycott = bool(_match(_BOYCOTT_SHOP, seq)) and bool(pairs & _BOYCOTT_ACT)
    # violence against a group / its places: intent or obligation forms (incl. "break" for places)
    # a bare infinitive is a slogan-style command when aimed at a group ("gini thiyanna!")
    en_gidx = [i for i in range(n) if raw[i] in _EN_GROUP]

    def group_pred(p):
        i, k, st, vi = p
        if k == "en":
            return any(abs(vi - g) <= 3 for g in en_gidx)
        if not (st in ("int", "obl") or (st == "inf" and k in _CALL_KINDS | {"break"})):
            return False
        if k == "hit":                                   # "demalunta gahanna ona": group right before verb
            return any(0 < vi - g <= 2 and sid[g] == sid[vi] for g in gidx)
        return near_group(vi)

    group_viol = any(group_pred(p) for p in live) and not benign
    en_hostile = bool({w for w in raw if w in _EN_HOSTILE})
    hate = None
    if slur and not mention and not (counter and not (expel or dehum or group_viol)):
        hate = "group_slur"
    elif para_group or koti_group:
        hate = "group_slur"
    elif (group or place) and not counter:
        if group_viol:
            hate = "group_violence"
        elif expel and group:
            hate = "group_expulsion"
        elif dehum and group:
            hate = "group_dehumanise"
        elif boycott and group:
            hate = "group_boycott"
        elif group in _EN_GROUP and en_hostile:
            hate = "group_violence" if {"kill", "killed", "murder", "murdered"} & set(raw) else "group_dehumanise"
    s.update(hate=hate, slur=slur, group=group or place)

    # ── personal abuse ──
    hard_idx = [i for i in range(n) if _HARD.match(seq[i])]
    soft_idx = [i for i in range(n) if _SOFT.match(seq[i])]
    animal_idx = [i for i in range(n) if _ANIMAL.match(seq[i])]
    # a mild word describing a THING ("moda wada", "balu wada" = dirty work) is not a name-call
    soft_person = [i for i in soft_idx if not (i + 1 < n and seq[i + 1] in _THING)]
    animal_person = [i for i in animal_idx if not (i + 1 < n and seq[i + 1] in _THING)]
    person_idx = [i for i in range(n) if seq[i] in _PERSON or raw[i] in _PERSON]
    # endearment is a CONSTRUCTION: "mage [chooti] buru patiya", "ane mage moda kolla" —
    # an affection marker just before the rough word, and a child/pet term just after it
    child_idx = [i for i in range(n) if seq[i] in _CHILD or raw[i] in _CHILD]
    endear_idx = [i for i in range(n) if seq[i] in _ENDEAR or raw[i] in _ENDEAR]
    endear = any(any(0 < w - e <= 2 for e in endear_idx) and any(0 < c - w <= 2 for c in child_idx)
                 for w in hard_idx + soft_idx)

    def near(a, b, d=2):
        return any(abs(x - y) <= d for x in a for y in b)

    s["vulgar"] = _match(_VULGAR, seq) or _match(_VULGAR, raw) or \
        ("[censored]" if _CENSORED.search(_deobf(text)) else None) or \
        ("🖕" if "\U0001F595" in str(text) else None) or \
        next((raw[i] for i in range(n - 1) if raw[i] in {"vesige", "wesige", "vestige", "wesiye"}
              and raw[i + 1].startswith(("puta", "putha"))), None)
    # "what a hypocrite!" — the insult noun must END the phrase ("what a clown show" is about a thing)
    what_a = any(raw[i] == "what" and raw[i + 1] in {"a", "an"} and _HARD.match(seq[i + 2]) and
                 (i + 3 >= n or sid[i + 3] != sid[i + 2])
                 for i in range(n - 2))
    s["hard"] = seq[hard_idx[0]] if hard_idx else None
    s["soft"] = seq[soft_idx[0]] if soft_idx else (seq[animal_idx[0]] if animal_idx else None)
    s["target"] = target
    s["nom_target"] = [raw[i] for i in nom_idx]
    s["negated"] = negated_any
    # an insult noun right next to "you" (umba modaya, තෝ බල්ලෙක්, you idiot) — or a mild
    # adjective right next to "you"/a person noun (umba moda, gon gani) — outside an
    # affectionate frame ("mage moda kolla")
    s["name_call"] = (not negated_any and not endear and what_a) or (not negated_any and not endear and
                      (near(hard_idx, nom_idx) or near(soft_person, nom_idx, 1) or
                       near(soft_person, person_idx, 1) or near(animal_person, nom_idx, 1)))
    s["benign"] = benign
    s["has_violence_verb"] = bool(preds)
    s["marker"] = marker
    s["endear"] = endear
    s["friendly"] = sorted(tokset & _FRIENDLY)
    s["casual"] = sorted(tokset & _CASUAL)
    s["ntok"] = n
    s["hostile_imp"] = bool(tokset & _HOSTILE_IMP)
    s["all_viol_negated"] = bool(preds) and not live           # every violence verb is negated
    s["counter"] = counter
    s["supportive"] = bool(tokset & _SUPPORT) and not preds
    s["festival"] = bool(tokset & _FESTIVAL) and not preds
    # every target is a POSSESSOR of a literal object ("oyage konde", "umbe cake eka")
    s["literal_possessive"] = bool(target) and all(
        (seq[i] in _GENITIVE or raw[i] in _GENITIVE) and i + 1 < n and seq[i + 1] in _BENIGN
        for i in tgt_idx)
    s["figurative_dying"] = bool(tokset & _DYING) and bool(tokset & _DYING_CTX) and not preds
    s["soft_thing"] = any(i + 1 < n and seq[i + 1] in _THING for i in soft_idx)
    return s


# ════════════════════════════════════════════════════════════════════════════
# 8. Verdict
# ════════════════════════════════════════════════════════════════════════════
RESCUE_MAX_P = 0.80     # a weak benign cue never overturns a confident model …
SHORT_TEXT = 8          # … except a SHORT romanized greeting/chat with no abusive cue at all


def _force(res, floor, rule):
    res = dict(res)
    res["offensive_score"] = max(res["offensive_score"], floor)
    res["label"] = OFF
    res["confidence"] = res["offensive_score"]
    res["probabilities"] = {NOT: 1.0 - res["offensive_score"], OFF: res["offensive_score"]}
    res["rule"] = rule
    return res, True


def _rescue(res, rule):
    res = dict(res)
    res["offensive_score"] = min(res["offensive_score"], 0.20)
    res["label"] = NOT
    res["confidence"] = 1.0 - res["offensive_score"]
    res["probabilities"] = {NOT: res["confidence"], OFF: res["offensive_score"]}
    res["rule"] = rule
    return res, True


def apply_safety_net(text, res, signals=None):
    """Combine the model verdict with the linguistic layer. Returns (res, changed).
    UPGRADES (-> Offensive) cover harms the model is blind to (threats, curses, identity
    hate, obscenity, name-calling). RESCUES (-> Not offensive) fix its known false positives
    on casual chat, endearment and literal action verbs — narrowly, and never when any
    abusive cue is present."""
    s = signals or analyse(text)
    res = dict(res)
    res.setdefault("rule", None)

    # ── UPGRADES ──
    if s["threat"] in ("threat_target", "threat_implied"):
        return _force(res, 0.93, s["threat"])
    if s["curse"]:
        return _force(res, 0.92, "curse")
    if s["hate"]:
        return _force(res, 0.92, s["hate"])
    if s["threat"] == "violence_call":
        return _force(res, 0.90, "violence_call")
    if s["vulgar"]:
        return _force(res, 0.92, "vulgar")
    if s["name_call"]:
        return _force(res, 0.90, "name_call")

    # ── RESCUES ── (also resolve the "uncertain" band 0.42-0.62 when a benign construction is recognised)
    p = res["offensive_score"]
    if p < 0.42:
        return res, False
    abusive = s["vulgar"] or s["hard"] or s["curse"] or s["hate"] or s["threat"] or s["name_call"]
    if not abusive and p < RESCUE_MAX_P:
        if s["all_viol_negated"]:
            return _rescue(res, "negated")               # "mama umbawa maranne na" — I won't kill you
        if s["group"] and s["counter"] and not s["slur"]:
            return _rescue(res, "counter_speech")        # "demala minissunta gahanna epa" — anti-hate
        if s["supportive"]:
            return _rescue(res, "supportive")            # "mama umbawa beragannam" — I'll save you
    if not abusive and s["festival"] and (p < RESCUE_MAX_P or s["ntok"] <= SHORT_TEXT) and not s["hostile_imp"]:
        return _rescue(res, "festival")                  # "vesak day", "Eid mubarak" — a religious observance
    if s["hard"] or s["group"] or s["negated"]:
        return res, False
    if s["endear"]:
        return _rescue(res, "endearment")               # "mage moda kolla" — a specific construction
    if s["has_violence_verb"] and s["benign"] and (not s["target"] or s["literal_possessive"]) \
            and p < RESCUE_MAX_P:
        return _rescue(res, "literal_verb")             # "konde kapanna yanawa" — a haircut
    if s["figurative_dying"] and (p < RESCUE_MAX_P or s["ntok"] <= SHORT_TEXT):
        return _rescue(res, "figurative")               # "hina wela marenawa" — dying of laughter
    if script_dominant(text):
        return res, False                                # remaining rescues are romanized-only
    short = s["ntok"] <= SHORT_TEXT
    if s["hostile_imp"]:
        return res, False                                # "palayan" / "wahapan" — never rescued
    if s["soft"] and s["soft_thing"] and not s["target"] and p < RESCUE_MAX_P:
        return _rescue(res, "soft_thing")               # "moda wada karanna epa" — a silly act
    if (s["friendly"] or s["casual"]) and not s["soft"] and (short or p < RESCUE_MAX_P):
        return _rescue(res, "casual")                   # "umba kohomada" — how are you
    return res, False


# ════════════════════════════════════════════════════════════════════════════
# 9. Plain-language reason (always consistent with the verdict above)
# ════════════════════════════════════════════════════════════════════════════
_REASON = {
    "threat_target": "It threatens violence against a person (e.g. “I'll kill/cut/hit you”) — "
                     "a direct threat is offensive even without a swear word.",
    "threat_implied": "It is a threat of violence. Sinhala often drops the pronoun, but the "
                      "intimidation (“know this”, “watch out”, “wait”) or the chained violent verbs "
                      "(“cut and kill”) make the target clear.",
    "violence_call": "It calls for someone to be killed or harmed (“… must be killed”).",
    "curse": "It is a death wish or curse aimed at someone (“go die”, “may lightning strike you”).",
    "group_slur": "It uses a derogatory slur for an ethnic or religious group — identity-based hate.",
    "group_violence": "It calls for violence against an ethnic or religious group or its places of worship — hate speech.",
    "group_expulsion": "It says an ethnic or religious group should be driven out or has no place in the country — hate speech.",
    "group_dehumanise": "It dehumanises or stereotypes an ethnic or religious group (animals, thieves, “breeding”) — hate speech.",
    "group_boycott": "It urges people to shun a group's businesses — a common hate trope (e.g. the 2018 “wanda pethi” rumour).",
    "vulgar": "Contains an explicit obscenity, which is abusive in any context.",
    "name_call": "It name-calls a person with an insulting noun — a direct personal insult.",
    "endearment": "The rough word sits inside an affectionate frame (“mage … patiya”), so it reads as endearment.",
    "literal_verb": "The action verb is used literally (food, hair, a call, a power cut), not against a person.",
    "figurative": "“Dying” is used figuratively (of laughter, hunger or tiredness) — not a threat or a death wish.",
    "negated": "The harmful verb is negated (“won't …”, “don't …”), so it is not a threat.",
    "counter_speech": "It mentions a group but argues AGAINST hate or violence — counter-speech, not hate speech.",
    "supportive": "It offers help or care to the person (save / help / look after), not harm.",
    "festival": "It mentions a religious or cultural observance (Vesak, Eid, Christmas, Pongal…) with no abusive "
                "cue — a known over-flagging bias of the model, corrected here.",
    "soft_thing": "The mild word describes an action or thing, not a person — not a personal insult.",
    "casual": "Casual, friendly address with no abusive word — reads as friendly banter.",
}


def context_reason(text, res, signals=None):
    s = signals or analyse(text)
    rule = res.get("rule")
    if rule in _REASON:
        return _REASON[rule]
    if 0.42 <= res["offensive_score"] <= 0.62:
        return ("The wording is borderline — it carries both neutral and hostile cues, so the "
                "system abstains and recommends human review.")
    if res["label"] == OFF:
        if s["hard"] and s["target"]:
            return "An insulting word is aimed at a person."
        return "The overall wording reads as hostile or abusive from its context."
    if s["negated"]:
        return "The harmful verb is negated (“won't …”, “don't …”), so it is not a threat."
    if s["group"] and _match(_COUNTER, s["seq"]):
        return "It mentions a group but argues against hate — counter-speech, not hate speech."
    if s["benign"] and not s["target"]:
        return "Any action verb here is literal (food, objects, services), not aimed at a person."
    if s["endear"]:
        return "A rough word inside an affectionate frame reads as endearment."
    if s["friendly"]:
        return "Casual friendly address with no abusive word — friendly banter."
    return "No slur, threat or hostile framing is present; the model reads it as ordinary language."


# ════════════════════════════════════════════════════════════════════════════
# 10. Evidence — the exact words behind a rule decision (shown in the UI)
# ════════════════════════════════════════════════════════════════════════════
def evidence(text, res, signals=None):
    """Return the words that triggered the rule decision in `res` (empty if the model decided)."""
    rule = res.get("rule")
    if not rule:
        return []
    s = signals or analyse(text)
    raw, seq = s["raw"], s["seq"]
    n = len(seq)
    ev = []

    def add(i):
        if 0 <= i < n and raw[i] not in ev:
            ev.append(raw[i])

    if rule.startswith("threat") or rule == "violence_call" or rule == "group_violence":
        for (i, k, st, vi) in _violence_preds(raw, seq):
            if not _negated(i, vi, raw, seq):
                add(i); add(vi)
        for i in range(n):
            if seq[i] in _TARGET or raw[i] in _TARGET or raw[i] in _EN_TARGET:
                add(i)
    if rule == "curse" or rule.startswith("threat"):
        allsets = _CURSE_TOK | _CURSE_PAIR | _MARKER
        for L in (1, 2, 3):
            for i in range(n - L + 1):
                if "".join(seq[i:i + L]) in allsets or "".join(raw[i:i + L]) in allsets:
                    for j in range(i, i + L):
                        add(j)
        for i in range(n):
            if seq[i] in {"hena", "henama", "හෙන", "හෙනම"}:
                add(i - 1); add(i); add(i + 1)
    if rule.startswith("group"):
        for i in range(n):
            if (_SLUR.match(seq[i]) or _GROUP.match(seq[i]) or _PLACE.match(seq[i]) or
                    _EXPEL_TOK.match(seq[i]) or _DEHUM.match(seq[i]) or raw[i] in _EN_GROUP or
                    raw[i] in _EN_HOSTILE or seq[i] in _PARA or seq[i] in _KOTI):
                add(i)
    if rule == "vulgar":
        for i in range(n):
            if _VULGAR.match(seq[i]) or _VULGAR.match(raw[i]):
                add(i)
    if rule == "name_call":
        for i in range(n):
            if (_HARD.match(seq[i]) or _SOFT.match(seq[i]) or _ANIMAL.match(seq[i]) or seq[i] in _NOM_TARGET or
                    raw[i] in _NOM_TARGET or seq[i] in _PERSON):
                add(i)
    return ev
