import math
import json
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from functools import lru_cache

import streamlit as st

# =========================
# Estilo (aprox al Kivy)
# =========================
UI = {
    "r_card": 18,
    "r_btn": 16,
    "h_btn_primary": 48,
    "h_btn_secondary": 40,
    "h_chip": 38,
    "sp_4": 4,
    "sp_8": 8,
    "sp_12": 12,
    "sp_16": 16,
    "sp_24": 24,
}

S = {
    "header_bg": "rgba(38,38,38,1)",
    "page_bg": "rgba(246,246,246,1)",
    "page_bg_top": "rgba(251,251,251,1)",
    "card_bg": "rgba(255,255,255,1)",
    "shadow": "rgba(0,0,0,0.10)",

    "text_dark": "rgba(31,31,31,1)",
    "text_mid": "rgba(97,97,97,1)",
    "text_light": "rgba(255,255,255,1)",

    "neon": "rgba(204,255,51,1)",
    "blue": "rgba(51,140,255,1)",
    "red": "rgba(242,77,89,1)",
    "green": "rgba(64,217,115,1)",

    "chip_off": "rgba(235,235,235,1)",
    "dark_btn": "rgba(59,59,59,1)",
    "loss_dot": "rgba(51,51,51,1)",
}

def inject_css():
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            background: linear-gradient(180deg, {S["page_bg_top"]} 0%, {S["page_bg"]} 55%, {S["page_bg"]} 100%) !important;
        }}

        /* quitar padding extra */
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }}

        /* Header */
        .ts-header {{
            background: {S["header_bg"]};
            border-radius: 16px;
            padding: 14px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: {S["text_light"]};
            margin-bottom: 14px;
            box-shadow: 0 10px 22px rgba(0,0,0,.12);
        }}
        .ts-title {{
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.2px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        /* Card */
        .ts-card {{
            background: {S["card_bg"]};
            border-radius: {UI["r_card"]}px;
            padding: 14px 16px;
            box-shadow: 0 10px 20px {S["shadow"]};
            margin-bottom: 12px;
            border: 1px solid rgba(0,0,0,0.04);
        }}
        .ts-card h3 {{
            margin: 0 0 8px 0;
            color: {S["text_dark"]};
            font-size: 15px;
        }}
        .muted {{
            color: {S["text_mid"]};
        }}

        /* Pills (Finish) */
        .pill {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 650;
            font-size: 13px;
            border: 1px solid rgba(0,0,0,0.06);
            background: {S["chip_off"]};
            color: {S["text_dark"]};
            margin: 4px 6px 0 0;
        }}
        .pill-on {{
            background: {S["neon"]};
            color: rgba(0,0,0,1);
        }}

        /* Ring */
        .ring-wrap {{
            background: {S["header_bg"]};
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 22px rgba(0,0,0,.10);
            margin-bottom: 12px;
        }}
        .ring-grid {{
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }}
        @media (max-width: 760px) {{
            .ring-grid {{ grid-template-columns: 1fr; }}
        }}

        .ring {{
            display:flex;
            flex-direction:column;
            align-items:center;
            gap: 8px;
            padding: 10px 10px;
            border-radius: 16px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .ring-title {{
            color: rgba(240,240,240,0.95);
            font-size: 13px;
            text-align:center;
            line-height: 1.2;
            white-space: pre-line;
        }}
        .ring-circle {{
            width: 98px;
            height: 98px;
            border-radius: 50%;
            display:flex;
            align-items:center;
            justify-content:center;
            background:
                conic-gradient({S["neon"]} var(--p), rgba(255,255,255,0.12) 0);
        }}
        .ring-inner {{
            width: 82px;
            height: 82px;
            border-radius: 50%;
            background: rgba(0,0,0,0.25);
            display:flex;
            align-items:center;
            justify-content:center;
            flex-direction:column;
        }}
        .ring-big {{
            color: {S["text_light"]};
            font-weight: 800;
            font-size: 18px;
            line-height:1.0;
        }}
        .ring-sub {{
            color: rgba(230,230,230,0.92);
            font-size: 11px;
            line-height:1.0;
            margin-top: 2px;
        }}

        /* Dots */
        .dots {{
            display:flex;
            gap: 10px;
            padding-top: 4px;
        }}
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: {S["loss_dot"]};
        }}
        .dot-win {{
            background: {S["neon"]};
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# =========================
# Lógica tenis
# =========================
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
def _prob_match_bo3(p_rounded: float, sets_me: int, sets_opp: int,
                    g_me: int, g_opp: int, pts_me: int, pts_opp: int, in_tb: bool) -> float:
    if sets_me >= 2:
        return 1.0
    if sets_opp >= 2:
        return 0.0

    p_set = _prob_set_from(p_rounded, g_me, g_opp, pts_me, pts_opp, in_tb)
    win_state = (p_rounded, sets_me + 1, sets_opp, 0, 0, 0, 0, False)
    lose_state = (p_rounded, sets_me, sets_opp + 1, 0, 0, 0, 0, False)

    return p_set * _prob_match_bo3(*win_state) + (1 - p_set) * _prob_match_bo3(*lose_state)

# =========================
# Estado live (igual que Kivy)
# =========================
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
        st, n, surf = self._undo.pop()
        self.state = st
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
        stt = self.state
        return _prob_match_bo3(p_r, stt.sets_me, stt.sets_opp, stt.games_me, stt.games_opp, stt.pts_me, stt.pts_opp, stt.in_tiebreak)

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

        self.points.append({
            "result": result,
            **meta,
            "surface": self.surface,
            "before": before,
            "set_idx": set_idx,
            "pressure": is_pressure,
        })

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

        finishes = {"winner": 0, "unforced": 0, "forced": 0, "ace": 0, "double_fault": 0,
                    "opp_error": 0, "opp_winner": 0, "none": 0}

        pressure_total = sum(1 for p in self.points if p.get("pressure"))
        pressure_won = sum(1 for p in self.points if p.get("pressure") and p["result"] == "win")

        for p in self.points:
            f = p.get("finish") or "none"
            if f not in finishes:
                finishes["none"] += 1
            else:
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

class MatchHistory:
    def __init__(self):
        self.matches = []

    def add(self, m: dict):
        self.matches.append(m)

    def filtered_matches(self, n=None, surface=None):
        arr = self.matches[:]
        if surface and surface != "Todas":
            arr = [m for m in arr if m.get("surface") == surface]
        if n is not None and n > 0:
            arr = arr[-n:]
        return arr

    def last_n_results(self, n=10, surface=None):
        matches = self.filtered_matches(n=n, surface=surface)
        return [("W" if m["won_match"] else "L") for m in matches[-n:]]

    def best_streak(self, surface=None):
        matches = self.filtered_matches(n=None, surface=surface)
        best = 0
        cur = 0
        for m in matches:
            if m["won_match"]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def pct(self, wins, total):
        return (wins / total * 100.0) if total else 0.0

    def aggregate(self, n=None, surface=None):
        matches = self.filtered_matches(n=n, surface=surface)

        total_m = len(matches)
        win_m = sum(1 for m in matches if m["won_match"])

        sets_w = sum(m["sets_w"] for m in matches)
        sets_l = sum(m["sets_l"] for m in matches)
        games_w = sum(m["games_w"] for m in matches)
        games_l = sum(m["games_l"] for m in matches)

        surfaces = {}
        for m in matches:
            srf = m["surface"]
            surfaces.setdefault(srf, {"w": 0, "t": 0})
            surfaces[srf]["t"] += 1
            if m["won_match"]:
                surfaces[srf]["w"] += 1

        points_total = sum(m.get("points_total", 0) for m in matches)
        points_won = sum(m.get("points_won", 0) for m in matches)
        pressure_total = sum(m.get("pressure_total", 0) for m in matches)
        pressure_won = sum(m.get("pressure_won", 0) for m in matches)

        finishes_sum = {"winner": 0, "unforced": 0, "forced": 0, "ace": 0, "double_fault": 0,
                        "opp_error": 0, "opp_winner": 0}
        for m in matches:
            fin = m.get("finishes", {}) or {}
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

# =========================
# Helpers UI web
# =========================
def header(title: str):
    st.markdown(
        f"""
        <div class="ts-header">
            <div class="ts-title">🎾 <span>{title}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

def card(title: str, body_html: str = ""):
    st.markdown(
        f"""
        <div class="ts-card">
            <h3>{title}</h3>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def ring(title: str, pct: float, subtitle: str):
    p = int(round(max(0, min(100, pct))))
    return f"""
    <div class="ring">
        <div class="ring-circle" style="--p:{p}%;">
            <div class="ring-inner">
                <div class="ring-big">{p}%</div>
                <div class="ring-sub">{subtitle}</div>
            </div>
        </div>
        <div class="ring-title">{title}</div>
    </div>
    """

def dots_html(results):
    if not results:
        return '<div class="muted">Aún no hay partidos guardados.</div>'
    bits = []
    for r in results:
        cls = "dot dot-win" if r == "W" else "dot"
        bits.append(f'<div class="{cls}"></div>')
    return f'<div class="dots">{"".join(bits)}</div>'

def pill_html(text: str, selected: bool):
    cls = "pill pill-on" if selected else "pill"
    return f'<span class="{cls}">{text}</span>'

# =========================
# Session State init
# =========================
def init_state():
    if "live" not in st.session_state:
        st.session_state.live = LiveMatch()
    if "history" not in st.session_state:
        st.session_state.history = MatchHistory()
    if "page" not in st.session_state:
        st.session_state.page = "LIVE"
    if "finish" not in st.session_state:
        st.session_state.finish = None
    if "filter_n" not in st.session_state:
        st.session_state.filter_n = 10
    if "filter_surface" not in st.session_state:
        st.session_state.filter_surface = "Todas"
    if "show_finish_form" not in st.session_state:
        st.session_state.show_finish_form = False

# =========================
# Pages
# =========================
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

def page_live():
    header("LIVE MATCH")

    live: LiveMatch = st.session_state.live

    # Top controls (Surface + nav)
    c1, c2, c3 = st.columns([1.2, 0.9, 0.9], vertical_alignment="center")
    with c1:
        surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(live.surface), key="surface_sel")
        if surface != live.surface:
            live.surface = surface

    with c2:
        if st.button("📊 Analysis", use_container_width=True):
            st.session_state.page = "ANALYSIS"
            st.rerun()
    with c3:
        if st.button("📈 Stats", use_container_width=True):
            st.session_state.page = "STATS"
            st.rerun()

    # Score card
    stt = live.state
    total, won, pct = live.points_stats()
    pts = f"TB {stt.pts_me}-{stt.pts_opp}" if stt.in_tiebreak else game_point_label(stt.pts_me, stt.pts_opp)
    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card(
        "Marcador",
        f"""
        <div style="font-size:22px;font-weight:800;color:{S["text_dark"]};margin-bottom:6px;">
          Sets {stt.sets_me}-{stt.sets_opp} &nbsp; · &nbsp; Juegos {stt.games_me}-{stt.games_opp} &nbsp; · &nbsp; Puntos {pts}
        </div>
        <div class="muted" style="margin-bottom:4px;">
          Superficie: <b>{live.surface}</b> &nbsp; · &nbsp; Puntos: <b>{total}</b> &nbsp; · &nbsp; % ganados: <b>{pct:.1f}%</b>
        </div>
        <div class="muted">
          Modelo: p(punto)≈<b>{p_point:.2f}</b> &nbsp; · &nbsp; Win Prob≈<b>{p_match:.1f}%</b>
        </div>
        """
    )

    # Point buttons
    st.markdown('<div class="ts-card"><h3>Punto</h3>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    with b2:
        if st.button("Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()

    g1, g2, s1, s2 = st.columns(4)
    with g1:
        if st.button("+Juego Yo", use_container_width=True):
            live.add_game_manual("me")
            st.rerun()
    with g2:
        if st.button("+Juego Rival", use_container_width=True):
            live.add_game_manual("opp")
            st.rerun()
    with s1:
        if st.button("+Set Yo", use_container_width=True):
            live.add_set_manual("me")
            st.rerun()
    with s2:
        if st.button("+Set Rival", use_container_width=True):
            live.add_set_manual("opp")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Finish chips
    st.markdown('<div class="ts-card"><h3>Finish (opcional)</h3>', unsafe_allow_html=True)

    # “visual” pills + buttons debajo para activar
    pills = []
    for key, label in FINISH_ITEMS:
        pills.append(pill_html(label, st.session_state.finish == key))
    st.markdown("".join(pills), unsafe_allow_html=True)

    # botones de selección en grid 3 cols (funcionalidad)
    cA, cB, cC = st.columns(3)
    for i, (key, label) in enumerate(FINISH_ITEMS):
        col = [cA, cB, cC][i % 3]
        with col:
            if st.button(label, key=f"finish_btn_{key}", use_container_width=True):
                st.session_state.finish = None if st.session_state.finish == key else key
                st.rerun()

    if st.button("Limpiar", use_container_width=False):
        st.session_state.finish = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Actions
    st.markdown('<div class="ts-card"><h3>Acciones</h3>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Deshacer", use_container_width=True):
            live.undo()
            st.rerun()
    with a2:
        if st.button("Analysis", use_container_width=True):
            st.session_state.page = "ANALYSIS"
            st.rerun()
    with a3:
        if st.button("Finalizar", use_container_width=True):
            st.session_state.show_finish_form = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Finalize form (modal simple)
    if st.session_state.show_finish_form:
        st.markdown('<div class="ts-card"><h3>Finalizar partido</h3>', unsafe_allow_html=True)
        with st.form("finish_form", clear_on_submit=False):
            colL, colR = st.columns(2)
            with colL:
                sets_w = st.number_input("Sets Yo", min_value=0, value=int(stt.sets_me), step=1)
                games_w = st.number_input("Juegos Yo", min_value=0, value=int(stt.games_me), step=1)
            with colR:
                sets_l = st.number_input("Sets Rival", min_value=0, value=int(stt.sets_opp), step=1)
                games_l = st.number_input("Juegos Rival", min_value=0, value=int(stt.games_opp), step=1)

            surface_fin = st.selectbox("Superficie (partido)", ("Tierra batida", "Pista rápida", "Hierba", "Indoor"), index=("Tierra batida", "Pista rápida", "Hierba", "Indoor").index(live.surface))
            cX, cY = st.columns(2)
            cancel = cX.form_submit_button("Cancelar")
            save = cY.form_submit_button("Guardar")

            if cancel:
                st.session_state.show_finish_form = False
                st.rerun()

            if save:
                won_match = int(sets_w) > int(sets_l)
                report = live.match_summary()
                st.session_state.history.add({
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "won_match": won_match,
                    "sets_w": int(sets_w), "sets_l": int(sets_l),
                    "games_w": int(games_w), "games_l": int(games_l),
                    "surface": surface_fin,
                    **report,
                })
                live.surface = surface_fin
                live.reset()
                st.session_state.finish = None
                st.session_state.show_finish_form = False
                st.success("Partido guardado ✅")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Export (por si lo quieres guardar)
    st.markdown('<div class="ts-card"><h3>Exportar</h3>', unsafe_allow_html=True)
    data = {"matches": st.session_state.history.matches}
    st.download_button(
        "Descargar historial (JSON)",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="tennisstats_history.json",
        mime="application/json"
    )
    st.markdown("</div>", unsafe_allow_html=True)

def page_analysis():
    header("Analysis")

    live: LiveMatch = st.session_state.live

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card(
        "Win Probability (modelo real)",
        f"""
        <div class="muted">p(punto)≈<b>{p_point:.2f}</b> &nbsp; · &nbsp; Win Prob≈<b>{p_match:.1f}%</b></div>
        <div class="muted" style="margin-top:8px;">
          Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.
        </div>
        """
    )

    probs = live.win_prob_series()
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        st.line_chart(probs, height=320)

    # Pressure stats
    pressure_total = sum(1 for p in live.points if p.get("pressure"))
    pressure_won = sum(1 for p in live.points if p.get("pressure") and p["result"] == "win")
    pressure_pct = (pressure_won / pressure_total * 100.0) if pressure_total else 0.0
    card(
        "Puntos de presión (live)",
        f"""
        <div class="muted"><b>{pressure_won}</b>/{pressure_total} ganados ({pressure_pct:.0f}%) en deuce/tiebreak.</div>
        """
    )

    if st.button("⬅️ Volver a LIVE", use_container_width=False):
        st.session_state.page = "LIVE"
        st.rerun()

def page_stats():
    header("Estadísticas")

    history: MatchHistory = st.session_state.history

    # Header panel (rings + filtros)
    st.markdown('<div class="ring-wrap">', unsafe_allow_html=True)

    agg = history.aggregate(n=st.session_state.filter_n, surface=st.session_state.filter_surface)

    rings_html = (
        '<div class="ring-grid">'
        + ring("Sets\nGanados", agg["sets_pct"], f'{agg["sets_w"]} de {agg["sets_w"] + agg["sets_l"]}')
        + ring("Partidos\nganados", agg["matches_pct"], f'{agg["matches_win"]} de {agg["matches_total"]}')
        + ring("Juegos\nGanados", agg["games_pct"], f'{agg["games_w"]} de {agg["games_w"] + agg["games_l"]}')
        + "</div>"
    )
    st.markdown(rings_html, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([1,1,1,1.2], vertical_alignment="center")
    with f1:
        if st.button("Últ. 10", use_container_width=True):
            st.session_state.filter_n = 10
            st.rerun()
    with f2:
        if st.button("Últ. 30", use_container_width=True):
            st.session_state.filter_n = 30
            st.rerun()
    with f3:
        if st.button("Todos", use_container_width=True):
            st.session_state.filter_n = None
            st.rerun()
    with f4:
        surf = st.selectbox("Superficie", ("Todas",) + SURFACES, index=(("Todas",) + SURFACES).index(st.session_state.filter_surface))
        if surf != st.session_state.filter_surface:
            st.session_state.filter_surface = surf
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Resumen
    card(
        "Resumen (filtro actual)",
        f"""
        <div class="muted">
            Puntos: <b>{agg["points_won"]}</b>/<b>{agg["points_total"]}</b> ({agg["points_pct"]:.0f}%) &nbsp; · &nbsp;
            Presión: <b>{agg["pressure_won"]}</b>/<b>{agg["pressure_total"]}</b> ({agg["pressure_pct"]:.0f}%)
        </div>
        <div class="muted" style="margin-top:6px;">
            Winners <b>{agg["finishes_sum"]["winner"]}</b> · ENF <b>{agg["finishes_sum"]["unforced"]}</b> · EF <b>{agg["finishes_sum"]["forced"]}</b> ·
            Aces <b>{agg["finishes_sum"]["ace"]}</b> · Dobles faltas <b>{agg["finishes_sum"]["double_fault"]}</b>
        </div>
        """
    )

    # Racha últimos 10
    surface_filter = None if st.session_state.filter_surface == "Todas" else st.session_state.filter_surface
    results = history.last_n_results(10, surface=surface_filter)
    card("Racha Últimos 10 Partidos", dots_html(results))

    # Mejor racha
    best = history.best_streak(surface=surface_filter)
    card("Mejor Racha", f'<div class="muted"><b style="font-size:18px;">{best}</b> victorias seguidas</div>')

    # Superficies
    order = ["Tierra batida", "Pista rápida", "Hierba", "Indoor"]
    surf = agg["surfaces"]
    rows = []
    for srf in order:
        w = surf.get(srf, {}).get("w", 0)
        t_ = surf.get(srf, {}).get("t", 0)
        pct = (w / t_ * 100.0) if t_ else 0.0
        rows.append((f"{pct:.0f}%", f"Victorias en {srf}", f"{w} de {t_}"))
    card("Superficies", "")
    st.table(rows)

    if st.button("⬅️ Volver a LIVE", use_container_width=False):
        st.session_state.page = "LIVE"
        st.rerun()

# =========================
# App
# =========================
def main():
    st.set_page_config(page_title="TennisStats", page_icon="🎾", layout="centered")
    inject_css()
    init_state()

    # Router simple
    page = st.session_state.page
    if page == "LIVE":
        page_live()
    elif page == "ANALYSIS":
        page_analysis()
    else:
        page_stats()

if __name__ == "__main__":
    main()
