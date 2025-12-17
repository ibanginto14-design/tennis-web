# app.py
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import json

import pandas as pd
import streamlit as st


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="TennisStats",
    page_icon="🎾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# THEME / CSS (fix titles not visible + mobile padding)
# ==========================================================
CSS = """
<style>
/* Make room for our fixed top bar on mobile */
.block-container { padding-top: 74px !important; padding-bottom: 28px !important; }

/* Fixed top bar */
.ts-topbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 64px;
  background: #1f1f1f;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 6px 18px rgba(0,0,0,.14);
}
.ts-topbar .inner {
  width: min(980px, calc(100vw - 28px));
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.ts-title {
  color: #ffffff;
  font-weight: 800;
  letter-spacing: .6px;
  font-size: 18px;
  line-height: 1;
  display: flex;
  align-items: center;
  gap: 10px;
}
.ts-badge {
  background: #ccff33;
  color: #111;
  font-weight: 800;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 12px;
}

/* Cards */
.ts-card {
  background: #ffffff;
  border-radius: 18px;
  padding: 16px 16px 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,.08);
  border: 1px solid rgba(0,0,0,.04);
}
.ts-card-title {
  font-weight: 800;
  color: #171717;
  margin: 0 0 10px 0;
  font-size: 16px;
}
.ts-muted { color: rgba(0,0,0,.60); font-size: 13px; }

/* Buttons spacing */
div.stButton > button {
  border-radius: 14px !important;
  padding: 12px 14px !important;
}

/* Ring component */
.ts-ring-wrap {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}
@media (max-width: 520px){
  .ts-ring-wrap { grid-template-columns: 1fr; }
}
.ts-ring {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: center;
  justify-content: center;
  background: rgba(255,255,255,.06);
  border-radius: 18px;
  padding: 8px 10px;
}
.ts-ring-circle {
  width: 110px;
  height: 110px;
  border-radius: 999px;
  background: conic-gradient(#ccff33 calc(var(--p)*1%), rgba(255,255,255,.14) 0);
  display: grid;
  place-items: center;
}
.ts-ring-inner {
  width: 86px;
  height: 86px;
  border-radius: 999px;
  background: #1f1f1f;
  display: grid;
  place-items: center;
  text-align: center;
  padding: 6px;
}
.ts-ring-big {
  color: #fff;
  font-weight: 900;
  font-size: 18px;
  line-height: 1.1;
}
.ts-ring-sub {
  color: rgba(255,255,255,.70);
  font-size: 12px;
  line-height: 1.1;
}
.ts-ring-title {
  color: rgba(255,255,255,.88);
  font-weight: 800;
  font-size: 12px;
  text-align: center;
  white-space: pre-line;
}

/* Dark header area for Stats page */
.ts-stats-header {
  background: #1f1f1f;
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,.12);
}

/* Streamlit default header is fine; we do NOT hide it to avoid odd mobile offsets */
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ==========================================================
# Tennis logic (from your model)
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
# State
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
        self.points: list[dict] = []
        self.state = LiveState()
        self.surface = "Tierra batida"
        self._undo: list[tuple[LiveState, int, str]] = []

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
        return _prob_match_bo3(p_r, st_.sets_me, st_.sets_opp, st_.games_me, st_.games_opp, st_.pts_me, st_.pts_opp, st_.in_tiebreak)

    def win_prob_series(self) -> list[float]:
        probs: list[float] = []
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
                "finish": meta.get("finish"),
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
            "none": 0,
        }

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
        self.matches: list[dict] = []

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
# Session init
# ==========================================================
if "live" not in st.session_state:
    st.session_state.live = LiveMatch()
if "history" not in st.session_state:
    st.session_state.history = MatchHistory()
if "page" not in st.session_state:
    st.session_state.page = "LIVE"
if "finish" not in st.session_state:
    st.session_state.finish = None


# ==========================================================
# Helpers UI
# ==========================================================
def topbar(title: str):
    st.markdown(
        f"""
        <div class="ts-topbar">
          <div class="inner">
            <div class="ts-title">🎾 {title}</div>
            <div class="ts-badge">TennisStats</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str):
    st.markdown(
        f"""
        <div class="ts-card">
          <div class="ts-card-title">{title}</div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ring(title: str, pct: float, sub: str) -> str:
    p = max(0, min(100, int(round(pct))))
    return f"""
    <div class="ts-ring">
      <div class="ts-ring-circle" style="--p:{p};">
        <div class="ts-ring-inner">
          <div>
            <div class="ts-ring-big">{p}%</div>
            <div class="ts-ring-sub">{sub}</div>
          </div>
        </div>
      </div>
      <div class="ts-ring-title">{title}</div>
    </div>
    """


SURFACES = ("Tierra batida", "Pista rápida", "Hierba", "Indoor")
FINISH_OPTIONS = [
    ("winner", "Winner"),
    ("unforced", "ENF"),
    ("forced", "EF"),
    ("ace", "Ace"),
    ("double_fault", "Doble falta"),
    ("opp_error", "Error rival"),
    ("opp_winner", "Winner rival"),
]


def finish_label(key: str | None) -> str:
    if not key:
        return "Ninguno"
    for k, lab in FINISH_OPTIONS:
        if k == key:
            return lab
    return "Ninguno"


def build_history_df(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        fin = m.get("finishes", {}) or {}
        rows.append(
            {
                "Fecha": m.get("date", ""),
                "Superficie": m.get("surface", ""),
                "Resultado": "W" if m.get("won_match") else "L",
                "Sets": f"{m.get('sets_w',0)}-{m.get('sets_l',0)}",
                "Juegos": f"{m.get('games_w',0)}-{m.get('games_l',0)}",
                "Puntos %": round(float(m.get("points_pct", 0) or 0), 1),
                "Presión %": round(float(m.get("pressure_pct", 0) or 0), 1),
                "Winners": int(fin.get("winner", 0) or 0),
                "ENF": int(fin.get("unforced", 0) or 0),
                "EF": int(fin.get("forced", 0) or 0),
                "Aces": int(fin.get("ace", 0) or 0),
                "Dobles faltas": int(fin.get("double_fault", 0) or 0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.iloc[::-1].reset_index(drop=True)  # newest first
    return df


# ==========================================================
# Navigation (simple, mobile friendly)
# ==========================================================
# Keep titles visible: we render our own topbar, then buttons
page = st.session_state.page
topbar("LIVE MATCH" if page == "LIVE" else ("Analysis" if page == "ANALYSIS" else "Stats"))

nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("🎾 LIVE", use_container_width=True):
        st.session_state.page = "LIVE"
        st.rerun()
with nav_cols[1]:
    if st.button("📊 Analysis", use_container_width=True):
        st.session_state.page = "ANALYSIS"
        st.rerun()
with nav_cols[2]:
    if st.button("📈 Stats", use_container_width=True):
        st.session_state.page = "STATS"
        st.rerun()

st.write("")


# ==========================================================
# LIVE PAGE
# ==========================================================
if st.session_state.page == "LIVE":
    live: LiveMatch = st.session_state.live
    history: MatchHistory = st.session_state.history

    # Surface
    surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(live.surface))
    live.surface = surface

    # Score card
    st_ = live.state
    total, won, pct = live.points_stats()
    pts_lbl = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card(
        "Marcador",
        f"""
        <div style="font-size:26px; font-weight:900; color:#111; line-height:1.15;">
          Sets {st_.sets_me}-{st_.sets_opp} &nbsp;·&nbsp; Juegos {st_.games_me}-{st_.games_opp} &nbsp;·&nbsp; Puntos {pts_lbl}
        </div>
        <div class="ts-muted" style="margin-top:8px;">
          Superficie: {live.surface} &nbsp;&nbsp;·&nbsp;&nbsp; Puntos: {total} &nbsp;&nbsp;·&nbsp;&nbsp; % ganados: {pct:.1f}%
        </div>
        <div class="ts-muted" style="margin-top:6px;">
          Modelo: p(punto)≈{p_point:.2f} &nbsp;&nbsp;·&nbsp;&nbsp; Win Prob≈{p_match:.1f}%
        </div>
        """,
    )
    st.write("")

    # Punto
    card("Punto", "<div class='ts-muted'>Registra el resultado del punto.</div>")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    with c2:
        if st.button("❌ Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        if st.button("+ Juego Yo", use_container_width=True):
            live.add_game_manual("me")
            st.rerun()
    with c4:
        if st.button("+ Juego Rival", use_container_width=True):
            live.add_game_manual("opp")
            st.rerun()
    with c5:
        if st.button("+ Set Yo", use_container_width=True):
            live.add_set_manual("me")
            st.rerun()
    with c6:
        if st.button("+ Set Rival", use_container_width=True):
            live.add_set_manual("opp")
            st.rerun()

    st.write("")

    # Finish (NO duplication anymore)
    card("Finish (opcional)", "<div class='ts-muted'>Selecciona (una sola vez) el tipo de finalización del punto.</div>")
    finish_keys = [None] + [k for k, _ in FINISH_OPTIONS]
    finish_labels = ["Ninguno"] + [lab for _, lab in FINISH_OPTIONS]

    # radio: one control only (no repeated buttons list)
    chosen = st.radio(
        "Finish",
        options=list(range(len(finish_keys))),
        format_func=lambda i: finish_labels[i],
        horizontal=True,
        index=finish_labels.index(finish_label(st.session_state.finish)),
        label_visibility="collapsed",
    )
    st.session_state.finish = finish_keys[chosen]

    cfin1, cfin2 = st.columns([1, 1])
    with cfin1:
        st.caption(f"Seleccionado: **{finish_label(st.session_state.finish)}**")
    with cfin2:
        if st.button("Limpiar finish", use_container_width=True):
            st.session_state.finish = None
            st.rerun()

    st.write("")

    # Actions
    card("Acciones", "<div class='ts-muted'>Deshacer, finalizar y exportación.</div>")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("↩️ Deshacer", use_container_width=True):
            live.undo()
            st.rerun()
    with a2:
        if st.button("📊 Ir a Analysis", use_container_width=True):
            st.session_state.page = "ANALYSIS"
            st.rerun()
    with a3:
        if st.button("🏁 Finalizar", use_container_width=True):
            st.session_state["show_finish_modal"] = True
            st.rerun()

    # Finish modal (simple inline)
    if st.session_state.get("show_finish_modal", False):
        st.write("")
        card("Finalizar partido", "<div class='ts-muted'>Introduce el resultado final y guarda en historial.</div>")

        sw = st.number_input("Sets Yo", min_value=0, value=int(live.state.sets_me), step=1)
        sl = st.number_input("Sets Rival", min_value=0, value=int(live.state.sets_opp), step=1)
        gw = st.number_input("Juegos Yo", min_value=0, value=int(live.state.games_me), step=1)
        gl = st.number_input("Juegos Rival", min_value=0, value=int(live.state.games_opp), step=1)
        surf_save = st.selectbox("Superficie (guardar)", SURFACES, index=SURFACES.index(live.surface))

        bsave, bcancel = st.columns(2)
        with bsave:
            if st.button("✅ Guardar partido", use_container_width=True):
                report = live.match_summary()
                won_match = sw > sl
                history.add(
                    {
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "won_match": bool(won_match),
                        "sets_w": int(sw),
                        "sets_l": int(sl),
                        "games_w": int(gw),
                        "games_l": int(gl),
                        "surface": surf_save,
                        **report,
                    }
                )
                live.surface = surf_save
                live.reset()
                st.session_state["show_finish_modal"] = False
                st.rerun()
        with bcancel:
            if st.button("Cancelar", use_container_width=True):
                st.session_state["show_finish_modal"] = False
                st.rerun()

    # Export section (now includes history preview)
    st.write("")
    card("Exportar", "<div class='ts-muted'>Descarga y vista previa del historial.</div>")

    df_hist = build_history_df(history.matches)
    if df_hist.empty:
        st.info("Aún no hay partidos guardados.")
    else:
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    # Downloads
    payload = {"matches": history.matches}
    st.download_button(
        "⬇️ Descargar historial (JSON)",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True,
    )
    if not df_hist.empty:
        st.download_button(
            "⬇️ Descargar historial (CSV)",
            data=df_hist.to_csv(index=False).encode("utf-8"),
            file_name="tennis_history.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==========================================================
# ANALYSIS PAGE
# ==========================================================
elif st.session_state.page == "ANALYSIS":
    live: LiveMatch = st.session_state.live

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0
    card(
        "Win Probability (modelo real)",
        f"""
        <div style="font-size:18px; font-weight:900; color:#111;">
          p(punto)≈{p_point:.2f} &nbsp;·&nbsp; Win Prob≈{p_match:.1f}%
        </div>
        <div class="ts-muted" style="margin-top:6px;">
          Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.
        </div>
        """,
    )
    st.write("")

    probs = live.win_prob_series()
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        df = pd.DataFrame({"Punto": list(range(1, len(probs) + 1)), "WinProb%": probs})
        st.line_chart(df.set_index("Punto"))

    # Pressure
    pressure_total = sum(1 for p in live.points if p.get("pressure"))
    pressure_won = sum(1 for p in live.points if p.get("pressure") and p["result"] == "win")
    pressure_pct = (pressure_won / pressure_total * 100.0) if pressure_total else 0.0
    st.write("")
    card(
        "Puntos de presión (live)",
        f"""
        <div style="font-size:16px; font-weight:900; color:#111;">
          {pressure_won}/{pressure_total} ganados ({pressure_pct:.0f}%)
        </div>
        <div class="ts-muted" style="margin-top:6px;">
          Se consideran presión los puntos en deuce y/o tiebreak.
        </div>
        """,
    )
    st.write("")
    if st.button("⬅️ Volver a LIVE", use_container_width=True):
        st.session_state.page = "LIVE"
        st.rerun()


# ==========================================================
# STATS PAGE
# ==========================================================
else:
    history: MatchHistory = st.session_state.history

    # Filters
    card("Filtros", "<div class='ts-muted'>Selecciona rango y superficie.</div>")
    f1, f2, f3 = st.columns(3)
    with f1:
        last10 = st.button("Últ. 10", use_container_width=True)
    with f2:
        last30 = st.button("Últ. 30", use_container_width=True)
    with f3:
        allm = st.button("Todos", use_container_width=True)

    if "filter_n" not in st.session_state:
        st.session_state.filter_n = 10
    if last10:
        st.session_state.filter_n = 10
        st.rerun()
    if last30:
        st.session_state.filter_n = 30
        st.rerun()
    if allm:
        st.session_state.filter_n = None
        st.rerun()

    surface_filter = st.selectbox("Superficie", ("Todas",) + SURFACES, index=0)
    st.write("")

    agg = history.aggregate(n=st.session_state.filter_n, surface=surface_filter)

    # Header rings (rendered properly)
    rings_html = f"""
    <div class="ts-stats-header">
      <div class="ts-ring-wrap">
        {ring("Sets\\nGanados", agg["sets_pct"], f"{agg['sets_w']} de {agg['sets_w'] + agg['sets_l']}")}
        {ring("Partidos\\nGanados", agg["matches_pct"], f"{agg['matches_win']} de {agg['matches_total']}")}
        {ring("Juegos\\nGanados", agg["games_pct"], f"{agg['games_w']} de {agg['games_w'] + agg['games_l']}")}
      </div>
    </div>
    """
    st.markdown(rings_html, unsafe_allow_html=True)
    st.write("")

    # Summary
    fin = agg["finishes_sum"]
    card(
        "Resumen (filtro actual)",
        f"""
        <div class="ts-muted">
          Puntos: {agg['points_won']}/{agg['points_total']} ({agg['points_pct']:.0f}%)
          &nbsp;&nbsp;·&nbsp;&nbsp;
          Presión: {agg['pressure_won']}/{agg['pressure_total']} ({agg['pressure_pct']:.0f}%)
        </div>
        <div class="ts-muted" style="margin-top:6px;">
          Winners {fin['winner']} · ENF {fin['unforced']} · EF {fin['forced']} · Aces {fin['ace']} · Dobles faltas {fin['double_fault']}
        </div>
        """,
    )
    st.write("")

    # Streak
    results = history.last_n_results(10, surface=(None if surface_filter == "Todas" else surface_filter))
    if not results:
        st.info("Aún no hay partidos guardados.")
    else:
        card("Racha Últimos 10 Partidos", "<div class='ts-muted'>" + " ".join(results) + "</div>")

    st.write("")
    best = history.best_streak(surface=(None if surface_filter == "Todas" else surface_filter))
    card("Mejor Racha", f"<div style='font-size:18px; font-weight:900; color:#111;'>{best} victorias seguidas</div>")

    st.write("")
    # Surfaces table
    order = list(SURFACES)
    surf = agg["surfaces"]

    rows = []
    for srf in order:
        w = surf.get(srf, {}).get("w", 0)
        t_ = surf.get(srf, {}).get("t", 0)
        pct_ = (w / t_ * 100.0) if t_ else 0.0
        rows.append({"Superficie": srf, "Victorias": w, "Total": t_, "%": round(pct_, 0)})

    df_s = pd.DataFrame(rows)
    card("Superficies", "<div class='ts-muted'>Victorias por superficie (según el filtro).</div>")
    st.dataframe(df_s, use_container_width=True, hide_index=True)

    st.write("")
    if st.button("⬅️ Volver a LIVE", use_container_width=True):
        st.session_state.page = "LIVE"
        st.rerun()
