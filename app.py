import os
import re
import json
import base64
import secrets
import hashlib
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from functools import lru_cache

import streamlit as st


# ==========================================================
# CONFIG + CSS (COMPACTO / MÓVIL)
# ==========================================================
st.set_page_config(page_title="TennisStats", page_icon="🎾", layout="centered")

COMPACT_CSS = """
<style>
/* Reduce márgenes generales */
.block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; max-width: 900px;}
/* Reduce espacios entre elementos */
div[data-testid="stVerticalBlock"] > div {gap: 0.6rem;}
/* Reduce espacio del header de Streamlit */
header[data-testid="stHeader"] {height: 0.6rem;}
/* Botones un poco más compactos */
.stButton>button {padding: 0.45rem 0.8rem; border-radius: 12px;}
/* Inputs compactos */
div[data-baseweb="input"] input {padding-top: 0.45rem; padding-bottom: 0.45rem;}
/* Chips simulados */
.small-note {color: rgba(0,0,0,0.55); font-size: 0.92rem; line-height: 1.25rem;}
.kpi {font-size: 1.05rem; font-weight: 700;}
.badge {display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; background: #f1f3f5; margin-right: .35rem; margin-bottom: .35rem;}
hr {margin: 0.55rem 0;}

/* ======= NAV "casillas" (no arriba) ======= */
.navwrap {margin: 0.4rem 0 0.8rem 0;}
.navtitle {font-weight: 900; font-size: 1.15rem; margin-bottom: .3rem;}
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)


# ==========================================================
# STORAGE (multi-usuario privado por fichero)
# ==========================================================
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
HIST_DIR = os.path.join(DATA_DIR, "histories")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HIST_DIR, exist_ok=True)


def safe_user_key(username: str) -> str:
    u = (username or "").strip().lower()
    u = re.sub(r"[^a-z0-9_\-\.]", "_", u)
    u = re.sub(r"_+", "_", u).strip("_")
    return u[:40] if u else ""


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64d(s: str) -> bytes:
    s = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("utf-8"))


def hash_pin(pin: str, salt_b: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt_b, 200_000)
    return _b64e(dk)


def load_users() -> dict:
    ensure_dirs()
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_users(users: dict) -> None:
    ensure_dirs()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def history_path_for(user_key: str) -> str:
    ensure_dirs()
    return os.path.join(HIST_DIR, f"history__{user_key}.json")


def load_history_from_disk(user_key: str) -> list:
    path = history_path_for(user_key)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        matches = obj.get("matches", [])
        return matches if isinstance(matches, list) else []
    except Exception:
        return []


def save_history_to_disk(user_key: str, matches: list) -> None:
    path = history_path_for(user_key)
    payload = {"matches": matches}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ==========================================================
# LÓGICA TENIS (MARCADOR)
# ==========================================================
POINT_LABELS = {0: "0", 1: "15", 2: "30", 3: "40"}


def game_point_label(p_me: int, p_opp: int) -> str:
    if p_me >= 3 and p_opp >= 3:
        if p_me == p_opp:
            return "40-40"
        if p_me == p_opp + 1:
            return "AD-40"
        if p_opp == p_me + 1:
            return "40-AD"
    return f"{POINT_LABELS.get(p_me, '40')}-{POINT_LABELS.get(p_opp, '40')}"


def won_game(p_me: int, p_opp: int) -> bool:
    return p_me >= 4 and (p_me - p_opp) >= 2


def won_tiebreak(p_me: int, p_opp: int) -> bool:
    return p_me >= 7 and (p_me - p_opp) >= 2


def is_set_over(g_me: int, g_opp: int) -> bool:
    if g_me >= 6 and (g_me - g_opp) >= 2:
        return True
    if g_me == 7 and g_opp == 6:
        return True
    return False


# ==========================================================
# MODELO REAL: Markov (punto→juego→set→BO3)
# ==========================================================
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@lru_cache(maxsize=None)
def _prob_game_from(p_rounded: float, a: int, b: int) -> float:
    p = max(1e-6, min(1 - 1e-6, float(p_rounded)))
    q = 1.0 - p

    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0

    if a >= 3 and b >= 3:
        deuce = (p * p) / (p * p + q * q)
        if a == b:
            return deuce
        if a == b + 1:
            return p * 1.0 + q * deuce
        if b == a + 1:
            return p * deuce + q * 0.0
        return deuce

    return p * _prob_game_from(p_rounded, a + 1, b) + q * _prob_game_from(p_rounded, a, b + 1)


@lru_cache(maxsize=None)
def _prob_tiebreak_from(p_rounded: float, a: int, b: int) -> float:
    p = max(1e-6, min(1 - 1e-6, float(p_rounded)))
    q = 1.0 - p

    if a >= 7 and a - b >= 2:
        return 1.0
    if b >= 7 and b - a >= 2:
        return 0.0

    if a >= 6 and b >= 6:
        deuce = (p * p) / (p * p + q * q)
        if a == b:
            return deuce
        if a == b + 1:
            return p * 1.0 + q * deuce
        if b == a + 1:
            return p * deuce + q * 0.0
        return deuce

    return p * _prob_tiebreak_from(p_rounded, a + 1, b) + q * _prob_tiebreak_from(p_rounded, a, b + 1)


@lru_cache(maxsize=None)
def _prob_set_from(p_rounded: float, g_me: int, g_opp: int, pts_me: int, pts_opp: int, in_tb: bool) -> float:
    if is_set_over(g_me, g_opp):
        return 1.0
    if is_set_over(g_opp, g_me):
        return 0.0

    if in_tb:
        return _prob_tiebreak_from(p_rounded, pts_me, pts_opp)

    p_game = _prob_game_from(p_rounded, pts_me, pts_opp)

    def after_game(next_g_me, next_g_opp):
        if next_g_me == 6 and next_g_opp == 6:
            return _prob_set_from(p_rounded, 6, 6, 0, 0, True)
        return _prob_set_from(p_rounded, next_g_me, next_g_opp, 0, 0, False)

    return p_game * after_game(g_me + 1, g_opp) + (1 - p_game) * after_game(g_me, g_opp + 1)


@lru_cache(maxsize=None)
def _prob_match_bo3(
    p_rounded: float,
    sets_me: int,
    sets_opp: int,
    g_me: int,
    g_opp: int,
    pts_me: int,
    pts_opp: int,
    in_tb: bool,
) -> float:
    if sets_me >= 2:
        return 1.0
    if sets_opp >= 2:
        return 0.0

    p_set = _prob_set_from(p_rounded, g_me, g_opp, pts_me, pts_opp, in_tb)
    win_state = (p_rounded, sets_me + 1, sets_opp, 0, 0, 0, 0, False)
    lose_state = (p_rounded, sets_me, sets_opp + 1, 0, 0, 0, 0, False)
    return p_set * _prob_match_bo3(*win_state) + (1 - p_set) * _prob_match_bo3(*lose_state)


# ==========================================================
# ESTADO LIVE
# ==========================================================
@dataclass
class LiveState:
    sets_me: int = 0
    sets_opp: int = 0
    games_me: int = 0
    games_opp: int = 0
    pts_me: int = 0
    pts_opp: int = 0
    in_tiebreak: bool = False


class LiveMatch:
    def __init__(self):
        self.points = []
        self.state = LiveState()
        self.surface = "Tierra batida"
        self._undo = []

    def snapshot(self):
        self._undo.append((deepcopy(self.state), len(self.points), self.surface))

    def undo(self):
        if not self._undo:
            return
        st_, n, surf = self._undo.pop()
        self.state = st_
        self.points = self.points[:n]
        self.surface = surf

    def reset(self):
        self.points = []
        self.state = LiveState()
        self._undo = []

    def points_stats(self):
        total = len(self.points)
        won = sum(1 for p in self.points if p["result"] == "win")
        pct = (won / total * 100.0) if total else 0.0
        return total, won, pct

    def estimate_point_win_prob(self) -> float:
        n = len(self.points)
        w = sum(1 for p in self.points if p["result"] == "win")
        p = (w + 1) / (n + 2) if n >= 0 else 0.5
        return _clamp01(p)

    def match_win_prob(self) -> float:
        p = self.estimate_point_win_prob()
        p_r = round(p, 3)
        st_ = self.state
        return _prob_match_bo3(
            p_r, st_.sets_me, st_.sets_opp, st_.games_me, st_.games_opp, st_.pts_me, st_.pts_opp, st_.in_tiebreak
        )

    def win_prob_series(self):
        probs = []
        tmp = LiveMatch()
        tmp.surface = self.surface
        for p in self.points:
            tmp.add_point(p["result"], {"finish": p.get("finish")})
            probs.append(tmp.match_win_prob() * 100.0)
        return probs

    def _maybe_start_tiebreak(self):
        if self.state.games_me == 6 and self.state.games_opp == 6:
            self.state.in_tiebreak = True
            self.state.pts_me = 0
            self.state.pts_opp = 0

    def _award_game_to_me(self):
        self.state.games_me += 1
        self.state.pts_me = 0
        self.state.pts_opp = 0
        self.state.in_tiebreak = False
        self._maybe_start_tiebreak()
        self._maybe_award_set()

    def _award_game_to_opp(self):
        self.state.games_opp += 1
        self.state.pts_me = 0
        self.state.pts_opp = 0
        self.state.in_tiebreak = False
        self._maybe_start_tiebreak()
        self._maybe_award_set()

    def _maybe_award_set(self):
        if is_set_over(self.state.games_me, self.state.games_opp):
            self.state.sets_me += 1
            self.state.games_me = 0
            self.state.games_opp = 0
            self.state.pts_me = 0
            self.state.pts_opp = 0
            self.state.in_tiebreak = False
        elif is_set_over(self.state.games_opp, self.state.games_me):
            self.state.sets_opp += 1
            self.state.games_me = 0
            self.state.games_opp = 0
            self.state.pts_me = 0
            self.state.pts_opp = 0
            self.state.in_tiebreak = False

    def add_point(self, result: str, meta: dict):
        self.snapshot()
        before = deepcopy(self.state)
        set_idx = before.sets_me + before.sets_opp + 1
        is_pressure = bool(before.in_tiebreak or (before.pts_me >= 3 and before.pts_opp >= 3))

        self.points.append(
            {
                "result": result,
                **meta,
                "surface": self.surface,
                "before": before.__dict__,
                "set_idx": set_idx,
                "pressure": is_pressure,
            }
        )

        if result == "win":
            self.state.pts_me += 1
        else:
            self.state.pts_opp += 1

        if self.state.in_tiebreak:
            if won_tiebreak(self.state.pts_me, self.state.pts_opp):
                self.state.games_me = 7
                self.state.games_opp = 6
                self._maybe_award_set()
            elif won_tiebreak(self.state.pts_opp, self.state.pts_me):
                self.state.games_opp = 7
                self.state.games_me = 6
                self._maybe_award_set()
            return

        if won_game(self.state.pts_me, self.state.pts_opp):
            self._award_game_to_me()
        elif won_game(self.state.pts_opp, self.state.pts_me):
            self._award_game_to_opp()

    def add_game_manual(self, who: str):
        self.snapshot()
        if who == "me":
            self._award_game_to_me()
        else:
            self._award_game_to_opp()

    def add_set_manual(self, who: str):
        self.snapshot()
        if who == "me":
            self.state.sets_me += 1
        else:
            self.state.sets_opp += 1
        self.state.games_me = 0
        self.state.games_opp = 0
        self.state.pts_me = 0
        self.state.pts_opp = 0
        self.state.in_tiebreak = False

    def match_summary(self):
        total = len(self.points)
        won = sum(1 for p in self.points if p["result"] == "win")
        pct = (won / total * 100.0) if total else 0.0

        finishes = {
            "winner": 0,
            "unforced": 0,
            "forced": 0,
            "ace": 0,
            "double_fault": 0,
            "opp_error": 0,
            "opp_winner": 0,
        }

        pressure_total = sum(1 for p in self.points if p.get("pressure"))
        pressure_won = sum(1 for p in self.points if p.get("pressure") and p.get("result") == "win")

        for p in self.points:
            f = p.get("finish")
            if f in finishes:
                finishes[f] += 1

        return {
            "points_total": total,
            "points_won": won,
            "points_pct": pct,
            "pressure_total": pressure_total,
            "pressure_won": pressure_won,
            "pressure_pct": (pressure_won / pressure_total * 100.0) if pressure_total else 0.0,
            "finishes": finishes,
        }


# ==========================================================
# HISTORIAL
# ==========================================================
class MatchHistory:
    def __init__(self):
        self.matches = []

    def add(self, m: dict):
        self.matches.append(m)

    def filtered_matches(self, n=None, surface=None):
        arr = list(self.matches)
        if surface and surface != "Todas":
            arr = [m for m in arr if m.get("surface") == surface]
        if n is not None and n > 0:
            arr = arr[-n:]
        return arr

    def last_n_results(self, n=10, surface=None):
        matches = self.filtered_matches(n=n, surface=surface)
        return [("W" if m.get("won_match") else "L") for m in matches[-n:]]

    def best_streak(self, surface=None):
        matches = self.filtered_matches(n=None, surface=surface)
        best = 0
        cur = 0
        for m in matches:
            if m.get("won_match"):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    @staticmethod
    def pct(wins, total):
        return (wins / total * 100.0) if total else 0.0

    def aggregate(self, n=None, surface=None):
        matches = self.filtered_matches(n=n, surface=surface)

        total_m = len(matches)
        win_m = sum(1 for m in matches if m.get("won_match"))

        sets_w = sum(int(m.get("sets_w", 0)) for m in matches)
        sets_l = sum(int(m.get("sets_l", 0)) for m in matches)
        games_w = sum(int(m.get("games_w", 0)) for m in matches)
        games_l = sum(int(m.get("games_l", 0)) for m in matches)

        surfaces = {}
        for m in matches:
            srf = m.get("surface", "Tierra batida")
            surfaces.setdefault(srf, {"w": 0, "t": 0})
            surfaces[srf]["t"] += 1
            if m.get("won_match"):
                surfaces[srf]["w"] += 1

        points_total = sum(int(m.get("points_total", 0)) for m in matches)
        points_won = sum(int(m.get("points_won", 0)) for m in matches)
        pressure_total = sum(int(m.get("pressure_total", 0)) for m in matches)
        pressure_won = sum(int(m.get("pressure_won", 0)) for m in matches)

        finishes_sum = {
            "winner": 0,
            "unforced": 0,
            "forced": 0,
            "ace": 0,
            "double_fault": 0,
            "opp_error": 0,
            "opp_winner": 0,
        }
        for m in matches:
            fin = (m.get("finishes") or {})
            for k in finishes_sum:
                finishes_sum[k] += int(fin.get(k, 0) or 0)

        return {
            "matches_total": total_m,
            "matches_win": win_m,
            "matches_pct": self.pct(win_m, total_m),
            "sets_w": sets_w,
            "sets_l": sets_l,
            "sets_pct": self.pct(sets_w, sets_w + sets_l),
            "games_w": games_w,
            "games_l": games_l,
            "games_pct": self.pct(games_w, games_w + games_l),
            "points_total": points_total,
            "points_won": points_won,
            "points_pct": self.pct(points_won, points_total),
            "pressure_total": pressure_total,
            "pressure_won": pressure_won,
            "pressure_pct": self.pct(pressure_won, pressure_total),
            "finishes_sum": finishes_sum,
            "surfaces": surfaces,
        }


# ==========================================================
# NUEVO: Resumen tipo entrenador (basado en estadísticas)
# ==========================================================
def coach_summary_from_match(m: dict) -> str:
    """Genera un resumen tipo entrenador usando SOLO stats del partido guardado."""
    won = bool(m.get("won_match"))
    res = "Victoria" if won else "Derrota"

    pts_total = int(m.get("points_total", 0) or 0)
    pts_won = int(m.get("points_won", 0) or 0)
    pts_pct = float(m.get("points_pct", 0) or 0)

    pressure_total = int(m.get("pressure_total", 0) or 0)
    pressure_won = int(m.get("pressure_won", 0) or 0)
    pressure_pct = float(m.get("pressure_pct", 0) or 0)

    fin = (m.get("finishes") or {})
    winners = int(fin.get("winner", 0) or 0)
    enf = int(fin.get("unforced", 0) or 0)
    ef = int(fin.get("forced", 0) or 0)
    aces = int(fin.get("ace", 0) or 0)
    df = int(fin.get("double_fault", 0) or 0)
    opp_err = int(fin.get("opp_error", 0) or 0)
    opp_w = int(fin.get("opp_winner", 0) or 0)

    # Heurísticas simples (legibles)
    strengths = []
    focus = []

    # Eficiencia puntos
    if pts_pct >= 55:
        strengths.append(f"dominaste el intercambio de puntos ({pts_pct:.0f}%).")
    elif pts_pct <= 45 and pts_total >= 10:
        focus.append(f"subir el % de puntos ganados ({pts_pct:.0f}%).")
    else:
        strengths.append(f"tu % de puntos estuvo equilibrado ({pts_pct:.0f}%).")

    # Presión
    if pressure_total >= 6:
        if pressure_pct >= 55:
            strengths.append(f"gestionaste muy bien la presión ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
        elif pressure_pct <= 45:
            focus.append(f"mejorar puntos de presión ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
        else:
            strengths.append(f"en presión estuviste parejo ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
    elif pressure_total > 0:
        strengths.append(f"en los pocos puntos de presión estuviste {pressure_won}/{pressure_total}.")
    else:
        strengths.append("hubo pocos puntos de presión registrados.")

    # Winners vs errores
    if winners >= max(5, enf + 2):
        strengths.append("generaste muchos winners y fuiste ofensivo cuando tocaba.")
    if enf >= max(5, winners + 2):
        focus.append("reducir errores no forzados (ENF) en momentos clave.")
    if df >= 3:
        focus.append("controlar dobles faltas (ritual de saque + margen).")
    if aces >= 3:
        strengths.append("el saque fue un arma (aces).")

    # Balance global
    if (enf + df) > (winners + aces) and pts_total >= 15:
        focus.append("buscar más margen: altura/profundidad y seleccionar mejor el riesgo.")
    if opp_err >= 5 and winners < 3:
        strengths.append("sacaste puntos provocando error del rival: buena consistencia.")

    # Plan de 3 claves
    plan = []
    if "reducir errores no forzados (ENF) en momentos clave." in focus:
        plan.append("1) Prioriza 2-3 pelotas de seguridad antes de acelerar (evita el 'todo o nada').")
    else:
        plan.append("1) Mantén tu patrón principal (consistencia + ataque cuando la bola sea clara).")

    if any("presión" in x for x in focus):
        plan.append("2) En deuce/tiebreak: rutina corta (respira, objetivo simple, juega al %).")
    else:
        plan.append("2) Sigue usando tu rutina en puntos importantes: claridad de objetivo por punto.")

    if any("dobles faltas" in x for x in focus):
        plan.append("3) Saque: 1º con dirección + 2º con más efecto/altura; mismo ritual siempre.")
    else:
        plan.append("3) Ajusta el saque según rival: alterna direcciones y busca el primer golpe tras el saque.")

    # Texto final
    s_txt = " ".join(strengths) if strengths else "buen partido en líneas generales."
    f_txt = " ".join(focus) if focus else "pocos puntos débiles claros: sigue consolidando lo que funcionó."

    return (
        f"**Resumen del entrenador ({res})**\n\n"
        f"- **Qué funcionó:** {s_txt}\n"
        f"- **Qué mejorar:** {f_txt}\n\n"
        f"**Claves para el próximo partido**\n"
        f"{plan[0]}\n{plan[1]}\n{plan[2]}\n\n"
        f"**Datos rápidos:** Puntos {pts_won}/{pts_total} ({pts_pct:.0f}%) · "
        f"Presión {pressure_won}/{pressure_total} ({pressure_pct:.0f}%) · "
        f"Winners {winners} · ENF {enf} · EF {ef} · Ace {aces} · DF {df}"
    )


# ==========================================================
# SESSION STATE INIT
# ==========================================================
def ss_init():
    if "live" not in st.session_state:
        st.session_state.live = LiveMatch()
    if "history" not in st.session_state:
        st.session_state.history = MatchHistory()
    if "finish" not in st.session_state:
        st.session_state.finish = None
    if "page" not in st.session_state:
        st.session_state.page = "LIVE"
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "auth_key" not in st.session_state:
        st.session_state.auth_key = None
    if "authed" not in st.session_state:
        st.session_state.authed = False


ss_init()

SURFACES = ("Tierra batida", "Pista rápida", "Hierba", "Indoor")
FINISH_ITEMS = [
    ("winner", "Winner"),
    ("unforced", "ENF"),
    ("forced", "EF"),
    ("ace", "Ace"),
    ("double_fault", "Doble falta"),
    ("opp_error", "Error rival"),
    ("opp_winner", "Winner rival"),
]


def small_note(txt: str):
    st.markdown(f"<div class='small-note'>{txt}</div>", unsafe_allow_html=True)


def title_h(txt: str):
    st.markdown(f"## {txt}")


def nav_tiles():
    st.markdown("<div class='navwrap'><div class='navtitle'>Página</div></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        if st.button("🎾 LIVE", use_container_width=True):
            st.session_state.page = "LIVE"
            st.rerun()
    with c2:
        if st.button("📈 Analysis", use_container_width=True):
            st.session_state.page = "ANALYSIS"
            st.rerun()
    with c3:
        if st.button("📊 Stats", use_container_width=True):
            st.session_state.page = "STATS"
            st.rerun()


# ==========================================================
# AUTH UI
# ==========================================================
def auth_block():
    st.title("🎾 TennisStats")
    st.caption("Acceso privado por usuario (cada uno ve su propio historial).")

    users = load_users()
    tab_login, tab_register = st.tabs(["🔑 Entrar", "🆕 Crear usuario"])

    with tab_login:
        u = st.text_input("Usuario", value="", placeholder="Ej: ruben")
        pin = st.text_input("PIN", value="", type="password", placeholder="4-12 dígitos")
        if st.button("Entrar", use_container_width=True):
            key = safe_user_key(u)
            if not key or key not in users:
                st.error("Usuario no existe.")
                return
            if not pin:
                st.error("Introduce el PIN.")
                return
            rec = users[key]
            try:
                salt = _b64d(rec["salt"])
                want = rec["hash"]
                got = hash_pin(pin, salt)
            except Exception:
                st.error("Error leyendo credenciales. (users.json corrupto?)")
                return
            if secrets.compare_digest(got, want):
                st.session_state.authed = True
                st.session_state.auth_user = rec.get("display", u.strip() or key)
                st.session_state.auth_key = key
                st.session_state.history.matches = load_history_from_disk(key)
                st.success("Acceso correcto ✅")
                st.rerun()
            else:
                st.error("PIN incorrecto.")

    with tab_register:
        new_u = st.text_input("Nuevo usuario", value="", placeholder="Solo letras/números (mejor corto)")
        new_pin = st.text_input("Nuevo PIN", value="", type="password", placeholder="4-12 dígitos")
        new_pin2 = st.text_input("Repite PIN", value="", type="password")
        if st.button("Crear usuario", use_container_width=True):
            key = safe_user_key(new_u)
            if not key:
                st.error("El usuario no puede estar vacío.")
                return
            if key in users:
                st.error("Ese usuario ya existe.")
                return
            if not (new_pin.isdigit() and 4 <= len(new_pin) <= 12):
                st.error("El PIN debe ser numérico (4 a 12 dígitos).")
                return
            if new_pin != new_pin2:
                st.error("Los PIN no coinciden.")
                return

            salt = os.urandom(16)
            rec = {
                "display": new_u.strip(),
                "salt": _b64e(salt),
                "hash": hash_pin(new_pin, salt),
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            users[key] = rec
            save_users(users)
            save_history_to_disk(key, [])
            st.success("Usuario creado ✅ Ya puedes entrar en la pestaña 'Entrar'.")


# ==========================================================
# MAIN: requiere login
# ==========================================================
if not st.session_state.authed:
    auth_block()
    st.stop()

live: LiveMatch = st.session_state.live
history: MatchHistory = st.session_state.history
user_key = st.session_state.auth_key
user_display = st.session_state.auth_user

topL, topR = st.columns([1.2, 0.8], gap="small")
with topL:
    st.markdown(f"**👤 Usuario:** `{user_display}`")
with topR:
    if st.button("🚪 Salir", use_container_width=True):
        st.session_state.authed = False
        st.session_state.auth_user = None
        st.session_state.auth_key = None
        st.session_state.page = "LIVE"
        st.session_state.finish = None
        st.rerun()

nav_tiles()
st.divider()


# ==========================================================
# PAGE: LIVE
# ==========================================================
if st.session_state.page == "LIVE":
    title_h("LIVE MATCH")

    colA, colB = st.columns([1.15, 1.0], gap="small")
    with colA:
        live.surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(live.surface))
    with colB:
        total, won, pct = live.points_stats()
        st.markdown(f"<div class='kpi'>Puntos: {total} · {pct:.0f}% ganados</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Marcador", anchor=False)
    st_ = live.state
    pts_label = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)
    st.write(f"**Sets {st_.sets_me}-{st_.sets_opp} · Juegos {st_.games_me}-{st_.games_opp} · Puntos {pts_label}**")

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0
    small_note(f"Modelo: p(punto)≈{p_point:.2f} · Win Prob≈{p_match:.1f}%")

    st.divider()

    st.subheader("Punto", anchor=False)
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if st.button("🟩 Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    with c2:
        if st.button("🟥 Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()

    c3, c4 = st.columns(2, gap="small")
    with c3:
        if st.button("➕ Juego Yo", use_container_width=True):
            live.add_game_manual("me")
            st.rerun()
        if st.button("➕ Set Yo", use_container_width=True):
            live.add_set_manual("me")
            st.rerun()
    with c4:
        if st.button("➕ Juego Rival", use_container_width=True):
            live.add_game_manual("opp")
            st.rerun()
        if st.button("➕ Set Rival", use_container_width=True):
            live.add_set_manual("opp")
            st.rerun()

    st.divider()

    st.subheader("Finish (opcional)", anchor=False)
    small_note("Selecciona 1 (se aplica al siguiente punto). Puedes deseleccionar tocando de nuevo.")

    fcols = st.columns(2, gap="small")
    for i, (key, label) in enumerate(FINISH_ITEMS):
        with fcols[i % 2]:
            selected = (st.session_state.finish == key)
            txt = f"✅ {label}" if selected else label
            if st.button(txt, key=f"finish_{key}", use_container_width=True):
                st.session_state.finish = None if selected else key
                st.rerun()

    colx, coly = st.columns([1, 1], gap="small")
    with colx:
        if st.button("🧼 Limpiar", use_container_width=True):
            st.session_state.finish = None
            st.rerun()
    with coly:
        small_note(f"Seleccionado: **{st.session_state.finish or '—'}**")

    st.divider()

    st.subheader("Acciones", anchor=False)
    a1, a2, a3 = st.columns(3, gap="small")
    with a1:
        if st.button("↩️ Deshacer", use_container_width=True):
            live.undo()
            st.rerun()
    with a2:
        if st.button("📈 Ir a Analysis", use_container_width=True):
            st.session_state.page = "ANALYSIS"
            st.rerun()
    with a3:
        if st.button("🏁 Finalizar", use_container_width=True):
            st.session_state._open_finish = True

    if st.session_state.get("_open_finish", False):
        with st.expander("Finalizar partido", expanded=True):
            st.write("Introduce el marcador final y guarda el partido.")
            sw = st.number_input("Sets Yo", 0, 5, value=int(live.state.sets_me), step=1)
            sl = st.number_input("Sets Rival", 0, 5, value=int(live.state.sets_opp), step=1)
            gw = st.number_input("Juegos Yo", 0, 50, value=int(live.state.games_me), step=1)
            gl = st.number_input("Juegos Rival", 0, 50, value=int(live.state.games_opp), step=1)
            surf_save = st.selectbox("Superficie (guardar)", SURFACES, index=SURACES.index(live.surface))

            s_left, s_right = st.columns(2, gap="small")
            with s_left:
                if st.button("Cancelar", use_container_width=True):
                    st.session_state._open_finish = False
                    st.rerun()
            with s_right:
                if st.button("Guardar partido", use_container_width=True):
                    won_match = (sw > sl)
                    report = live.match_summary()

                    history.add({
                        "id": f"m_{datetime.now().timestamp()}",
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "won_match": bool(won_match),
                        "sets_w": int(sw), "sets_l": int(sl),
                        "games_w": int(gw), "games_l": int(gl),
                        "surface": surf_save,
                        **report,
                    })

                    save_history_to_disk(user_key, history.matches)

                    live.surface = surf_save
                    live.reset()
                    st.session_state.finish = None
                    st.session_state._open_finish = False
                    st.success("Partido guardado ✅")
                    st.rerun()

    st.divider()

    st.subheader("Exportar", anchor=False)
    small_note("Tu historial privado (solo tu usuario). Puedes editar/borrar y exportar/importar en JSON.")

    if not history.matches:
        st.info("Aún no hay partidos guardados.")
    else:
        matches = list(reversed(history.matches))
        for idx, m in enumerate(matches):
            real_i = len(history.matches) - 1 - idx
            date = m.get("date", "")
            surf = m.get("surface", "—")
            res = "✅ W" if m.get("won_match") else "❌ L"
            score = f"{m.get('sets_w',0)}-{m.get('sets_l',0)} sets · {m.get('games_w',0)}-{m.get('games_l',0)} juegos"
            pts = f"{m.get('points_won',0)}/{m.get('points_total',0)} pts ({m.get('points_pct',0):.0f}%)"

            with st.expander(f"{res} · {score} · {surf} · {date}", expanded=False):
                st.write(f"**{score}**")
                small_note(f"{pts} · Presión: {m.get('pressure_won',0)}/{m.get('pressure_total',0)} ({m.get('pressure_pct',0):.0f}%)")

                fin = (m.get("finishes") or {})
                fin_line = f"Winners {fin.get('winner',0)} · ENF {fin.get('unforced',0)} · EF {fin.get('forced',0)} · Ace {fin.get('ace',0)} · DF {fin.get('double_fault',0)}"
                small_note(fin_line)

                # NUEVO: botón resumen entrenador (por partido)
                if st.button("🧠 Resumen tipo entrenador", key=f"coach_{m.get('id',real_i)}", use_container_width=True):
                    st.session_state._coach_open = True
                    st.session_state._coach_text = coach_summary_from_match(m)
                    st.rerun()

                e1, e2 = st.columns(2, gap="small")
                with e1:
                    if st.button("✏️ Editar", key=f"edit_btn_{m.get('id',real_i)}", use_container_width=True):
                        st.session_state._edit_index = real_i
                        st.session_state._edit_open = True
                        st.rerun()
                with e2:
                    if st.button("🗑️ Borrar", key=f"del_btn_{m.get('id',real_i)}", use_container_width=True):
                        history.matches.pop(real_i)
                        save_history_to_disk(user_key, history.matches)
                        st.success("Partido borrado.")
                        st.rerun()

        # NUEVO: panel resumen entrenador
        if st.session_state.get("_coach_open", False):
            with st.expander("🧠 Resumen del entrenador", expanded=True):
                st.markdown(st.session_state.get("_coach_text", ""))
                if st.button("Cerrar resumen", use_container_width=True):
                    st.session_state._coach_open = False
                    st.session_state._coach_text = ""
                    st.rerun()

        if st.session_state.get("_edit_open", False):
            i = st.session_state.get("_edit_index", None)
            if i is not None and 0 <= i < len(history.matches):
                m = history.matches[i]
                with st.expander("✏️ Editar partido", expanded=True):
                    st.write("Modifica los campos y guarda.")
                    col1, col2 = st.columns(2, gap="small")
                    with col1:
                        won_match = st.toggle("Victoria", value=bool(m.get("won_match", False)))
                        sets_w = st.number_input("Sets Yo", 0, 5, value=int(m.get("sets_w", 0)), step=1)
                        games_w = st.number_input("Juegos Yo", 0, 50, value=int(m.get("games_w", 0)), step=1)
                    with col2:
                        sets_l = st.number_input("Sets Rival", 0, 5, value=int(m.get("sets_l", 0)), step=1)
                        games_l = st.number_input("Juegos Rival", 0, 50, value=int(m.get("games_l", 0)), step=1)
                        surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(m.get("surface", SURFACES[0])))

                    date = st.text_input("Fecha (ISO)", value=str(m.get("date", "")))

                    bL, bR = st.columns(2, gap="small")
                    with bL:
                        if st.button("Cancelar edición", use_container_width=True):
                            st.session_state._edit_open = False
                            st.session_state._edit_index = None
                            st.rerun()
                    with bR:
                        if st.button("Guardar cambios", use_container_width=True):
                            m["won_match"] = bool(won_match)
                            m["sets_w"] = int(sets_w)
                            m["sets_l"] = int(sets_l)
                            m["games_w"] = int(games_w)
                            m["games_l"] = int(games_l)
                            m["surface"] = surface
                            m["date"] = date
                            history.matches[i] = m

                            save_history_to_disk(user_key, history.matches)

                            st.session_state._edit_open = False
                            st.session_state._edit_index = None
                            st.success("Cambios guardados ✅")
                            st.rerun()

    export_obj = {"matches": history.matches}
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Descargar historial (JSON)",
        data=export_json,
        file_name=f"tennis_history__{user_key}.json",
        mime="application/json",
        use_container_width=True,
    )

    up = st.file_uploader("⬆️ Importar historial (JSON)", type=["json"], label_visibility="visible")
    if up is not None:
        try:
            obj = json.loads(up.read().decode("utf-8"))
            matches = obj.get("matches", [])
            if not isinstance(matches, list):
                raise ValueError("Formato incorrecto: 'matches' debe ser una lista.")
            for mm in matches:
                if "id" not in mm:
                    mm["id"] = f"m_{datetime.now().timestamp()}"
            history.matches = matches
            save_history_to_disk(user_key, history.matches)
            st.success("Historial importado ✅")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo importar: {e}")


# ==========================================================
# PAGE: ANALYSIS
# ==========================================================
elif st.session_state.page == "ANALYSIS":
    title_h("Analysis")

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0
    st.write("**Win Probability (modelo real)**")
    small_note(f"p(punto)≈{p_point:.2f} · Win Prob≈{p_match:.1f}%")
    small_note("Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.")

    probs = live.win_prob_series()
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        st.line_chart(probs, height=260)

    st.divider()
    st.subheader("Puntos de presión (live)", anchor=False)
    pressure_total = sum(1 for p in live.points if p.get("pressure"))
    pressure_won = sum(1 for p in live.points if p.get("pressure") and p.get("result") == "win")
    pressure_pct = (pressure_won / pressure_total * 100.0) if pressure_total else 0.0
    st.write(f"**{pressure_won}/{pressure_total}** ganados ({pressure_pct:.0f}%) en deuce/tiebreak.")


# ==========================================================
# PAGE: STATS
# ==========================================================
else:
    title_h("Stats")

    colF1, colF2 = st.columns([1.1, 0.9], gap="small")
    with colF1:
        n_choice = st.selectbox("Rango", ["Últ. 10", "Últ. 30", "Todos"], index=0)
    with colF2:
        surf_filter = st.selectbox("Superficie", ["Todas", *SURFACES], index=0)

    n = 10 if n_choice == "Últ. 10" else (30 if n_choice == "Últ. 30" else None)
    agg = history.aggregate(n=n, surface=surf_filter)

    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        st.metric("Partidos", f"{agg['matches_pct']:.0f}%", f"{agg['matches_win']} / {agg['matches_total']}")
    with k2:
        st.metric("Sets", f"{agg['sets_pct']:.0f}%", f"{agg['sets_w']} / {agg['sets_w'] + agg['sets_l']}")
    with k3:
        st.metric("Juegos", f"{agg['games_pct']:.0f}%", f"{agg['games_w']} / {agg['games_w'] + agg['games_l']}")

    st.divider()

    st.subheader("Resumen", anchor=False)
    st.write(
        f"**Puntos:** {agg['points_won']}/{agg['points_total']} ({agg['points_pct']:.0f}%) · "
        f"**Presión:** {agg['pressure_won']}/{agg['pressure_total']} ({agg['pressure_pct']:.0f}%)"
    )
    fin = agg["finishes_sum"]
    small_note(
        f"Winners {fin['winner']} · ENF {fin['unforced']} · EF {fin['forced']} · "
        f"Aces {fin['ace']} · Dobles faltas {fin['double_fault']}"
    )

    st.divider()

    st.subheader("Racha últimos 10", anchor=False)
    results = history.last_n_results(10, surface=(None if surf_filter == "Todas" else surf_filter))
    if not results:
        st.info("Aún no hay partidos guardados.")
    else:
        row = []
        for r in results:
            row.append("✅ W" if r == "W" else "⬛ L")
        st.write(" · ".join(row))

    st.subheader("Mejor racha", anchor=False)
    best = history.best_streak(surface=(None if surf_filter == "Todas" else surf_filter))
    st.write(f"**{best}** victorias seguidas")

    st.divider()

    st.subheader("Superficies", anchor=False)
    order = list(SURFACES)
    surf = agg["surfaces"]
    for srf in order:
        w = surf.get(srf, {}).get("w", 0)
        t_ = surf.get(srf, {}).get("t", 0)
        pct = (w / t_ * 100.0) if t_ else 0.0
        st.write(f"**{srf}:** {pct:.0f}%  ({w} de {t_})")
