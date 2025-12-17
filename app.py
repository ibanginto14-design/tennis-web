# app.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st


# ==========================================================
# LÓGICA TENIS (igual que tu base)
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
    in_tb: bool
) -> float:
    if sets_me >= 2:
        return 1.0
    if sets_opp >= 2:
        return 0.0

    p_set = _prob_set_from(p_rounded, g_me, g_opp, pts_me, pts_opp, in_tb)

    win_state = (p_rounded, sets_me + 1, sets_opp, 0, 0, 0, 0, False)
    lose_state = (p_rounded, sets_me, sets_opp + 1, 0, 0, 0, 0, False)

    return p_set * _prob_match_bo3(*win_state) + (1 - p_set) * _prob_match_bo3(*lose_state)


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
        self.points: List[Dict[str, Any]] = []
        self.state = LiveState()
        self.surface = "Tierra batida"
        self._undo: List[Any] = []

    def snapshot(self):
        self._undo.append((deepcopy(self.state), len(self.points), self.surface))

    def undo(self):
        if not self._undo:
            return
        st0, n, surf = self._undo.pop()
        self.state = st0
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
        st0 = self.state
        return _prob_match_bo3(
            p_r,
            st0.sets_me,
            st0.sets_opp,
            st0.games_me,
            st0.games_opp,
            st0.pts_me,
            st0.pts_opp,
            st0.in_tiebreak
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

        self.points.append({
            "result": result,
            **meta,
            "surface": self.surface,
            "before": asdict(before),
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
            finishes[f if f in finishes else "none"] += 1

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
        self.matches: List[Dict[str, Any]] = []

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

    @staticmethod
    def pct(wins, total):
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


# ==========================================================
# UI (Streamlit)
# ==========================================================
def inject_css():
    st.markdown(
        """
        <style>
        /* Layout */
        .block-container { padding-top: 0.7rem; padding-bottom: 2rem; max-width: 980px; }
        @media (max-width: 480px) { .block-container { padding-left: 0.9rem; padding-right: 0.9rem; } }

        /* Header */
        .topbar {
            background: #222; color: #fff;
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 26px rgba(0,0,0,.12);
            display:flex; align-items:center; gap:10px;
        }
        .topbar .ball {
            width: 26px; height: 26px; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #d8ff55, #8ecb1a);
            box-shadow: inset 0 0 0 3px rgba(255,255,255,0.55);
        }
        .topbar h1 { font-size: 18px; margin:0; padding:0; letter-spacing: .04em; }

        /* Cards */
        .card {
            background:#fff;
            border-radius: 18px;
            padding: 14px 14px 10px 14px;
            box-shadow: 0 8px 20px rgba(0,0,0,.08);
            margin: 12px 0;
        }
        .card h3 { margin:0 0 10px 0; font-size: 15px; color:#222; }
        .muted { color:#666; font-size: 13px; }
        .big { font-size: 22px; font-weight: 800; color:#111; margin: 3px 0 8px 0; }

        /* Chips */
        .chiprow { display:flex; gap:10px; flex-wrap:wrap; }
        .chip {
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,.10);
            background: #efefef;
            color:#111;
            font-weight: 650;
            font-size: 13px;
            cursor:pointer;
            user-select:none;
        }
        .chip.sel {
            background: #d8ff55;
            border-color: rgba(0,0,0,.16);
        }

        /* Rings */
        .rings { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        @media (max-width: 480px) { .rings { grid-template-columns: 1fr; } }

        .ring {
            background: rgba(255,255,255,0.16);
            border-radius: 18px;
            padding: 10px 10px 12px 10px;
            box-shadow: 0 10px 24px rgba(0,0,0,.12);
        }
        .ring-title { color:#f2f2f2; font-weight: 700; font-size: 13px; margin-top: 8px; text-align:center; white-space: pre-line; }
        .ring-circle {
            width: 110px; height: 110px; border-radius: 50%;
            margin: 0 auto;
            background: conic-gradient(#d8ff55 calc(var(--p) * 1%), rgba(255,255,255,0.14) 0);
            display:flex; align-items:center; justify-content:center;
        }
        .ring-inner {
            width: 82px; height: 82px; border-radius: 50%;
            background: rgba(0,0,0,0.18);
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            color:#fff;
        }
        .ring-big { font-weight: 900; font-size: 18px; line-height: 1.1; }
        .ring-sub { font-size: 12px; opacity: .85; }

        /* Dots streak */
        .dots { display:flex; gap: 10px; align-items:center; padding: 6px 2px; flex-wrap:wrap; }
        .dot { width: 12px; height: 12px; border-radius: 50%; background:#222; opacity:.55; }
        .dot.w { background:#d8ff55; opacity:1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state():
    if "page" not in st.session_state:
        st.session_state.page = "live"
    if "live" not in st.session_state:
        st.session_state.live = LiveMatch()
    if "history" not in st.session_state:
        st.session_state.history = MatchHistory()
    if "finish_sel" not in st.session_state:
        st.session_state.finish_sel = None
    if "toast" not in st.session_state:
        st.session_state.toast = None


def header(title: str):
    st.markdown(
        f"""
        <div class="topbar">
            <div class="ball"></div>
            <h1>{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def nav_buttons():
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 Analysis", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()
    with c2:
        if st.button("📈 Stats", use_container_width=True):
            st.session_state.page = "stats"
            st.rerun()


def card_open(title: str):
    st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def ring(title: str, pct: float, sub: str):
    pct0 = int(round(max(0, min(100, pct))))
    st.markdown(
        f"""
        <div class="ring">
          <div class="ring-circle" style="--p:{pct0};">
            <div class="ring-inner">
              <div class="ring-big">{pct0}%</div>
              <div class="ring-sub">{sub}</div>
            </div>
          </div>
          <div class="ring-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def history_dataframe(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame(columns=["Fecha", "Resultado", "Sets", "Juegos", "Superficie", "% Puntos", "% Presión"])
    rows = []
    for m in matches:
        rows.append({
            "Fecha": m.get("date", ""),
            "Resultado": "W" if m.get("won_match") else "L",
            "Sets": f"{m.get('sets_w',0)}-{m.get('sets_l',0)}",
            "Juegos": f"{m.get('games_w',0)}-{m.get('games_l',0)}",
            "Superficie": m.get("surface", ""),
            "% Puntos": round(float(m.get("points_pct", 0.0)), 1),
            "% Presión": round(float(m.get("pressure_pct", 0.0)), 1),
        })
    df = pd.DataFrame(rows)
    return df


# ==========================================================
# PÁGINAS
# ==========================================================
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
    live: LiveMatch = st.session_state.live
    hist: MatchHistory = st.session_state.history

    header("LIVE MATCH")

    # Superficie + nav
    live.surface = st.selectbox(
        "Superficie",
        ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
        index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(live.surface)
        if live.surface in ["Tierra batida", "Pista rápida", "Hierba", "Indoor"] else 0,
    )
    nav_buttons()

    # Marcador
    st0 = live.state
    total, won, pct = live.points_stats()
    pts_txt = f"TB {st0.pts_me}-{st0.pts_opp}" if st0.in_tiebreak else game_point_label(st0.pts_me, st0.pts_opp)
    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card_open("Marcador")
    st.markdown(
        f'<div class="big">Sets {st0.sets_me}-{st0.sets_opp} &nbsp;·&nbsp; Juegos {st0.games_me}-{st0.games_opp} &nbsp;·&nbsp; Puntos {pts_txt}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="muted">Superficie: {live.surface} &nbsp;·&nbsp; Puntos: {total} &nbsp;·&nbsp; % ganados: {pct:.1f}%</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="muted">Modelo: p(punto)≈{p_point:.2f} &nbsp;·&nbsp; Win Prob≈{p_match:.1f}%</div>',
        unsafe_allow_html=True
    )
    card_close()

    # Punto
    card_open("Punto")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish_sel})
            st.session_state.finish_sel = None
            st.rerun()
    with c2:
        if st.button("Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish_sel})
            st.session_state.finish_sel = None
            st.rerun()

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        if st.button("+Juego Yo", use_container_width=True):
            live.add_game_manual("me")
            st.rerun()
    with r2:
        if st.button("+Juego Rival", use_container_width=True):
            live.add_game_manual("opp")
            st.rerun()
    with r3:
        if st.button("+Set Yo", use_container_width=True):
            live.add_set_manual("me")
            st.rerun()
    with r4:
        if st.button("+Set Rival", use_container_width=True):
            live.add_set_manual("opp")
            st.rerun()
    card_close()

    # Finish (opcional) ✅ YA NO SE DUPLICA
    card_open("Finish (opcional)")
    st.markdown('<div class="chiprow">', unsafe_allow_html=True)

    # Render chips como botones "simulados" (con state)
    # Usamos botones normales en columnas para que sea clickable + responsive
    cols = st.columns(4)
    for i, (k, label) in enumerate(FINISH_ITEMS):
        with cols[i % 4]:
            selected = (st.session_state.finish_sel == k)
            text = f"✅ {label}" if selected else label
            if st.button(text, key=f"chip_{k}", use_container_width=True):
                st.session_state.finish_sel = None if selected else k
                st.rerun()

    # Limpiar
    if st.button("Limpiar", use_container_width=False):
        st.session_state.finish_sel = None
        st.rerun()
    card_close()

    # Acciones
    card_open("Acciones")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Deshacer", use_container_width=True):
            live.undo()
            st.rerun()
    with a2:
        if st.button("Analysis", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()
    with a3:
        open_finish = st.button("Finalizar", use_container_width=True)
    card_close()

    # Finalizar (form)
    if open_finish:
        st.session_state._show_finish = True
        st.rerun()

    if st.session_state.get("_show_finish", False):
        card_open("Finalizar partido")
        sw = st.number_input("Sets Yo", min_value=0, value=int(st0.sets_me), step=1)
        sl = st.number_input("Sets Rival", min_value=0, value=int(st0.sets_opp), step=1)
        gw = st.number_input("Juegos Yo", min_value=0, value=int(st0.games_me), step=1)
        gl = st.number_input("Juegos Rival", min_value=0, value=int(st0.games_opp), step=1)
        surf_save = st.selectbox("Superficie (guardar)", ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
                                 index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(live.surface))

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Cancelar", use_container_width=True):
                st.session_state._show_finish = False
                st.rerun()
        with b2:
            if st.button("Guardar partido", use_container_width=True):
                report = live.match_summary()
                won_match = int(sw) > int(sl)

                hist.add({
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "won_match": won_match,
                    "sets_w": int(sw),
                    "sets_l": int(sl),
                    "games_w": int(gw),
                    "games_l": int(gl),
                    "surface": surf_save,
                    **report,
                })

                live.surface = surf_save
                live.reset()
                st.session_state.finish_sel = None
                st.session_state._show_finish = False
                st.success("Partido guardado ✅")
                st.rerun()
        card_close()

    # Exportar + HISTORIAL VISIBLE ✅
    card_open("Exportar")
    df = history_dataframe(hist.matches)
    if df.empty:
        st.info("Aún no hay partidos guardados.")
    else:
        st.write("Historial de partidos:")
        st.dataframe(df, use_container_width=True, hide_index=True)

    payload = {"matches": hist.matches}
    st.download_button(
        "Descargar historial (JSON)",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True,
    )

    if not df.empty:
        st.download_button(
            "Descargar historial (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="tennis_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    card_close()


def page_analysis():
    live: LiveMatch = st.session_state.live

    header("Analysis")
    if st.button("⬅️ Volver a LIVE", use_container_width=True):
        st.session_state.page = "live"
        st.rerun()

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card_open("Win Probability (modelo real)")
    st.markdown(
        f'<div class="big">p(punto)≈{p_point:.2f} &nbsp;·&nbsp; Win Prob≈{p_match:.1f}%</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.</div>',
        unsafe_allow_html=True
    )

    probs = live.win_prob_series()
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        chart_df = pd.DataFrame({"WinProb%": probs})
        st.line_chart(chart_df, height=320)
    card_close()

    card_open("Puntos de presión (live)")
    total = sum(1 for p in live.points if p.get("pressure"))
    won = sum(1 for p in live.points if p.get("pressure") and p["result"] == "win")
    pct = (won / total * 100.0) if total else 0.0
    st.markdown(f'<div class="muted">{won}/{total} ganados ({pct:.0f}%) en deuce/tiebreak.</div>', unsafe_allow_html=True)
    card_close()


def page_stats():
    hist: MatchHistory = st.session_state.history

    header("Estadísticas")
    if st.button("⬅️ Volver a LIVE", use_container_width=True):
        st.session_state.page = "live"
        st.rerun()

    # Filtros
    card_open("Filtros")
    fcols = st.columns([1, 1, 1, 1.2])
    with fcols[0]:
        if st.button("Últ. 10", use_container_width=True):
            st.session_state.filter_n = 10
    with fcols[1]:
        if st.button("Últ. 30", use_container_width=True):
            st.session_state.filter_n = 30
    with fcols[2]:
        if st.button("Todos", use_container_width=True):
            st.session_state.filter_n = None
    with fcols[3]:
        st.session_state.filter_surface = st.selectbox(
            "Superficie",
            ["Todas", "Tierra batida", "Pista rápida", "Hierba", "Indoor"],
            index=["Todas", "Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(
                st.session_state.get("filter_surface", "Todas")
            ),
        )
    card_close()

    n = st.session_state.get("filter_n", 10)
    surf = st.session_state.get("filter_surface", "Todas")

    agg = hist.aggregate(n=n, surface=surf)

    # Rings (header visual)
    st.markdown(
        '<div class="card" style="background:#222; color:#fff; margin-top:10px;">'
        '<div class="rings">',
        unsafe_allow_html=True
    )
    ring("Sets\nGanados", agg["sets_pct"], f"{agg['sets_w']} de {agg['sets_w'] + agg['sets_l']}")
    ring("Partidos\nganados", agg["matches_pct"], f"{agg['matches_win']} de {agg['matches_total']}")
    ring("Juegos\nGanados", agg["games_pct"], f"{agg['games_w']} de {agg['games_w'] + agg['games_l']}")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Resumen
    card_open("Resumen (filtro actual)")
    st.markdown(
        f'<div class="muted">Puntos: {agg["points_won"]}/{agg["points_total"]} ({agg["points_pct"]:.0f}%) '
        f'&nbsp;·&nbsp; Presión: {agg["pressure_won"]}/{agg["pressure_total"]} ({agg["pressure_pct"]:.0f}%)</div>',
        unsafe_allow_html=True
    )
    fin = agg["finishes_sum"]
    st.markdown(
        f'<div class="muted">Winners {fin["winner"]} &nbsp;·&nbsp; ENF {fin["unforced"]} &nbsp;·&nbsp; EF {fin["forced"]} '
        f'&nbsp;·&nbsp; Aces {fin["ace"]} &nbsp;·&nbsp; Dobles faltas {fin["double_fault"]}</div>',
        unsafe_allow_html=True
    )
    card_close()

    # Racha últimos 10
    card_open("Racha Últimos 10 Partidos")
    results = hist.last_n_results(10, surface=(None if surf == "Todas" else surf))
    if not results:
        st.markdown('<div class="muted">Aún no hay partidos guardados.</div>', unsafe_allow_html=True)
    else:
        dots_html = '<div class="dots">' + "".join(
            ['<span class="dot w"></span>' if r == "W" else '<span class="dot"></span>' for r in results]
        ) + "</div>"
        st.markdown(dots_html, unsafe_allow_html=True)
    card_close()

    # Mejor racha
    card_open("Mejor Racha")
    best = hist.best_streak(surface=(None if surf == "Todas" else surf))
    st.markdown(f'<div class="big">{best} victorias seguidas</div>', unsafe_allow_html=True)
    card_close()

    # Superficies
    card_open("Superficies")
    order = ["Tierra batida", "Pista rápida", "Hierba", "Indoor"]
    surf_map = agg["surfaces"]
    rows = []
    for srf in order:
        w = surf_map.get(srf, {}).get("w", 0)
        t_ = surf_map.get(srf, {}).get("t", 0)
        pct0 = (w / t_ * 100.0) if t_ else 0.0
        rows.append({"%": f"{pct0:.0f}%", "Etiqueta": f"Victorias en {srf}", "Cuenta": f"{w} de {t_}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    card_close()


# ==========================================================
# MAIN
# ==========================================================
def main():
    st.set_page_config(page_title="TennisStats", page_icon="🎾", layout="centered")
    ensure_state()
    inject_css()

    page = st.session_state.page
    if page == "analysis":
        page_analysis()
    elif page == "stats":
        page_stats()
    else:
        page_live()


if __name__ == "__main__":
    main()
