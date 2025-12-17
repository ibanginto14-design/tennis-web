# app.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from copy import deepcopy
from datetime import datetime
from functools import lru_cache

import streamlit as st


# ==========================================================
# CONFIG STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="TennisStats",
    page_icon="🎾",
    layout="centered",
)

# ==========================================================
# ESTILOS (muy importante para móvil + rings)
# ==========================================================
def inject_css():
    st.markdown(
        """
<style>
/* --- Layout base --- */
:root{
  --header-bg: #262626;
  --card-bg: #ffffff;
  --page-bg: #f6f6f6;
  --text-dark: #1f1f1f;
  --text-mid: #606060;
  --neon: #ccff33;
  --blue: #338cff;
  --red: #f24d59;
  --green: #40d973;
  --shadow: 0 10px 24px rgba(0,0,0,.08);
  --radius: 18px;
}

html, body, [class*="css"]{
  background: var(--page-bg) !important;
}

main .block-container{
  padding-top: 18px;
  padding-bottom: 32px;
  max-width: 980px;
}

/* Oculta elementos Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* --- Header tipo Kivy --- */
.ts-header{
  background: var(--header-bg);
  color: white;
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
  display:flex;
  align-items:center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 14px;
}

.ts-title{
  font-weight: 800;
  letter-spacing: .5px;
  display:flex;
  align-items:center;
  gap: 10px;
}

.ts-pill{
  display:inline-flex;
  align-items:center;
  gap:10px;
}

/* --- Cards --- */
.ts-card{
  background: var(--card-bg);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 16px 16px;
  margin: 12px 0;
}

.ts-card h3{
  margin: 0 0 10px 0;
  color: var(--text-dark);
  font-size: 16px;
}

.ts-muted{
  color: var(--text-mid);
  font-size: 13px;
  margin-top: 6px;
}

/* --- “chips” (finish) --- */
.ts-chip-row{
  display:flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 6px;
}
.ts-chip{
  border-radius: 999px;
  padding: 8px 12px;
  background: #eaeaea;
  color: #222;
  font-size: 13px;
  border: 1px solid rgba(0,0,0,.06);
}
.ts-chip.selected{
  background: var(--neon);
  color: #111;
}

/* --- Rings (IMPORTANT: esto evita que se vea el HTML como texto) --- */
.rings{
  display:grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 10px;
  margin-bottom: 8px;
}

.ring{
  display:flex;
  flex-direction: column;
  align-items:center;
  justify-content:center;
}

.ring-circle{
  --p: 0%;
  width: 92px;
  height: 92px;
  border-radius: 50%;
  background:
    conic-gradient(var(--neon) var(--p), rgba(255,255,255,.18) 0);
  position: relative;
  box-shadow: 0 10px 16px rgba(0,0,0,.10);
}

.ring-inner{
  position:absolute;
  inset: 12px;
  border-radius: 50%;
  background: var(--header-bg);
  display:flex;
  align-items:center;
  justify-content:center;
  flex-direction: column;
  color: #fff;
}

.ring-big{ font-weight: 800; font-size: 18px; line-height: 1; }
.ring-sub{ font-size: 11px; opacity: .85; margin-top: 4px; }
.ring-title{
  margin-top: 8px;
  text-align:center;
  font-size: 12px;
  color: #f2f2f2;
  opacity:.95;
}

/* --- Filters row responsive --- */
.ts-filters{
  display:flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items:center;
}

/* --- Mobile adjustments --- */
@media (max-width: 640px){
  main .block-container{ padding-left: 12px; padding-right: 12px; }
  .rings{ grid-template-columns: 1fr; }
  .ring-circle{ width: 110px; height: 110px; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ==========================================================
# Lógica tenis (idéntica a tu Kivy)
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


# ==========================================================
# Estado live / historial
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
        return _prob_match_bo3(p_r, st_.sets_me, st_.sets_opp, st_.games_me, st_.games_opp, st_.pts_me, st_.pts_opp, st_.in_tiebreak)

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
# Session state init
# ==========================================================
def ss_init():
    if "live" not in st.session_state:
        st.session_state.live = LiveMatch()
    if "history" not in st.session_state:
        st.session_state.history = MatchHistory()
    if "page" not in st.session_state:
        st.session_state.page = "live"
    if "finish" not in st.session_state:
        st.session_state.finish = None
    if "filter_n" not in st.session_state:
        st.session_state.filter_n = 10
    if "filter_surface" not in st.session_state:
        st.session_state.filter_surface = "Todas"


ss_init()


# ==========================================================
# UI helpers
# ==========================================================
def header(title: str):
    st.markdown(
        f"""
<div class="ts-header">
  <div class="ts-title">🎾 <span>{title}</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def card_open(title: str):
    st.markdown(f"""<div class="ts-card"><h3>{title}</h3>""", unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def ring_html(title: str, pct: float, subtitle: str) -> str:
    pct_i = int(round(max(0, min(100, pct))))
    return f"""
<div class="ring">
  <div class="ring-circle" style="--p:{pct_i}%;">
    <div class="ring-inner">
      <div class="ring-big">{pct_i}%</div>
      <div class="ring-sub">{subtitle}</div>
    </div>
  </div>
  <div class="ring-title">{title}</div>
</div>
    """


def go(page: str):
    st.session_state.page = page
    st.rerun()


# ==========================================================
# PAGES
# ==========================================================
def page_live():
    header("LIVE MATCH")

    live: LiveMatch = st.session_state.live

    # Top controls
    colA, colB = st.columns([1.2, 1.0])
    with colA:
        surface = st.selectbox(
            "Superficie",
            ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
            index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(live.surface),
        )
        live.surface = surface

    with colB:
        c1, c2 = st.columns(2)
        with c1:
            st.button("📊 Analysis", use_container_width=True, on_click=lambda: go("analysis"))
        with c2:
            st.button("📈 Stats", use_container_width=True, on_click=lambda: go("stats"))

    # Score card
    st_ = live.state
    total, won, pct_pts = live.points_stats()
    pts_lbl = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card_open("Marcador")
    st.markdown(
        f"**Sets {st_.sets_me}-{st_.sets_opp}**  ·  **Juegos {st_.games_me}-{st_.games_opp}**  ·  **Puntos {pts_lbl}**"
    )
    st.markdown(f"<div class='ts-muted'>Superficie: {live.surface}  ·  Puntos: {total}  ·  % ganados: {pct_pts:.1f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='ts-muted'>Modelo: p(punto)≈{p_point:.2f}  ·  Win Prob≈{p_match:.1f}%</div>", unsafe_allow_html=True)
    card_close()

    # Point card
    card_open("Punto")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    with c2:
        if st.button("Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        if st.button("+Juego Yo", use_container_width=True):
            live.add_game_manual("me"); st.rerun()
    with c4:
        if st.button("+Juego Rival", use_container_width=True):
            live.add_game_manual("opp"); st.rerun()
    with c5:
        if st.button("+Set Yo", use_container_width=True):
            live.add_set_manual("me"); st.rerun()
    with c6:
        if st.button("+Set Rival", use_container_width=True):
            live.add_set_manual("opp"); st.rerun()

    card_close()

    # Finish card (chips)
    card_open("Finish (opcional)")
    finish_items = [
        ("winner", "Winner"),
        ("unforced", "ENF"),
        ("forced", "EF"),
        ("ace", "Ace"),
        ("double_fault", "Doble falta"),
        ("opp_error", "Error rival"),
        ("opp_winner", "Winner rival"),
    ]

    # Render chips (visual)
    chips = []
    for k, label in finish_items:
        cls = "ts-chip selected" if st.session_state.finish == k else "ts-chip"
        chips.append(f"<span class='{cls}'>{label}</span>")
    st.markdown("<div class='ts-chip-row'>" + "".join(chips) + "</div>", unsafe_allow_html=True)

    # Controls (functional) in a grid of buttons
    bcols = st.columns(3)
    for i, (k, label) in enumerate(finish_items):
        with bcols[i % 3]:
            if st.button(label, use_container_width=True, key=f"finish_{k}"):
                st.session_state.finish = None if st.session_state.finish == k else k
                st.rerun()

    if st.button("Limpiar", use_container_width=False):
        st.session_state.finish = None
        st.rerun()
    card_close()

    # Actions
    card_open("Acciones")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Deshacer", use_container_width=True):
            live.undo(); st.rerun()
    with a2:
        st.button("Analysis", use_container_width=True, on_click=lambda: go("analysis"))
    with a3:
        st.session_state._show_finish_form = True

    card_close()

    # Finish form (simple + robust for Streamlit Cloud)
    with st.expander("Finalizar partido", expanded=bool(st.session_state.get("_show_finish_form", False))):
        sw = st.number_input("Sets Yo", min_value=0, step=1, value=int(st_.sets_me))
        sl = st.number_input("Sets Rival", min_value=0, step=1, value=int(st_.sets_opp))
        gw = st.number_input("Juegos Yo", min_value=0, step=1, value=int(st_.games_me))
        gl = st.number_input("Juegos Rival", min_value=0, step=1, value=int(st_.games_opp))
        srf = st.selectbox("Superficie (guardar)", ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
                           index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(live.surface))

        if st.button("Guardar partido", use_container_width=True):
            won_match = sw > sl
            report = live.match_summary()
            st.session_state.history.add({
                "date": datetime.now().isoformat(timespec="seconds"),
                "won_match": bool(won_match),
                "sets_w": int(sw), "sets_l": int(sl),
                "games_w": int(gw), "games_l": int(gl),
                "surface": srf,
                **report,
            })
            live.surface = srf
            live.reset()
            st.session_state.finish = None
            st.session_state._show_finish_form = False
            st.success("Partido guardado ✅")
            st.rerun()

    # Export JSON
    card_open("Exportar")
    data = {"matches": st.session_state.history.matches}
    st.download_button(
        "Descargar historial (JSON)",
        data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True,
    )
    card_close()


def page_analysis():
    header("Analysis")
    live: LiveMatch = st.session_state.live

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card_open("Win Probability (modelo real)")
    st.markdown(f"**p(punto)≈{p_point:.2f}**  ·  **Win Prob≈{p_match:.1f}%**")
    st.markdown(
        "<div class='ts-muted'>Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.</div>",
        unsafe_allow_html=True,
    )

    probs = live.win_prob_series()
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        # Gráfica simple, eje 0-100 (evita esos decimales raros que se ven en móvil)
        st.line_chart(probs, height=240)

    card_close()

    card_open("Puntos de presión (live)")
    total = sum(1 for p in live.points if p.get("pressure"))
    won = sum(1 for p in live.points if p.get("pressure") and p["result"] == "win")
    pct = (won / total * 100.0) if total else 0.0
    st.markdown(f"**{won}/{total}** ganados (**{pct:.0f}%**) en deuce/tiebreak.")
    card_close()

    st.button("⬅️ Volver a LIVE", use_container_width=True, on_click=lambda: go("live"))


def page_stats():
    header("Estadísticas")
    hist: MatchHistory = st.session_state.history

    # Rings header block
    agg = hist.aggregate(n=st.session_state.filter_n, surface=st.session_state.filter_surface)

    st.markdown(
        "<div class='ts-card' style='background: var(--header-bg); color:white;'>"
        "<h3 style='color:white;margin:0 0 8px 0;'>Ganados</h3>",
        unsafe_allow_html=True
    )

    # ✅ AQUI está la parte crítica: unsafe_allow_html=True para que NO se vea el HTML como texto.
    rings_html = (
        "<div class='rings'>"
        + ring_html("Sets<br>Ganados", agg["sets_pct"], f"{agg['sets_w']} de {agg['sets_w'] + agg['sets_l']}")
        + ring_html("Partidos<br>ganados", agg["matches_pct"], f"{agg['matches_win']} de {agg['matches_total']}")
        + ring_html("Juegos<br>Ganados", agg["games_pct"], f"{agg['games_w']} de {agg['games_w'] + agg['games_l']}")
        + "</div>"
    )
    st.markdown(rings_html, unsafe_allow_html=True)

    # Filters
    st.markdown("<div class='ts-filters'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        if st.button("Últ. 10", use_container_width=True):
            st.session_state.filter_n = 10
            st.rerun()
    with c2:
        if st.button("Últ. 30", use_container_width=True):
            st.session_state.filter_n = 30
            st.rerun()
    with c3:
        if st.button("Todos", use_container_width=True):
            st.session_state.filter_n = None
            st.rerun()
    with c4:
        st.session_state.filter_surface = st.selectbox(
            "Superficie",
            ["Todas", "Tierra batida", "Pista rápida", "Hierba", "Indoor"],
            index=["Todas", "Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(st.session_state.filter_surface),
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close header card

    # Summary cards
    card_open("Resumen (filtro actual)")
    st.markdown(
        f"<div class='ts-muted'>Puntos: {agg['points_won']}/{agg['points_total']} ({agg['points_pct']:.0f}%)  ·  "
        f"Presión: {agg['pressure_won']}/{agg['pressure_total']} ({agg['pressure_pct']:.0f}%)</div>",
        unsafe_allow_html=True
    )
    fin = agg["finishes_sum"]
    st.markdown(
        f"<div class='ts-muted'>Winners {fin['winner']}  ·  ENF {fin['unforced']}  ·  EF {fin['forced']}  ·  "
        f"Aces {fin['ace']}  ·  Dobles faltas {fin['double_fault']}</div>",
        unsafe_allow_html=True
    )
    card_close()

    card_open("Racha Últimos 10 Partidos")
    results = hist.last_n_results(10, surface=(None if st.session_state.filter_surface == "Todas" else st.session_state.filter_surface))
    if not results:
        st.write("Aún no hay partidos guardados.")
    else:
        # simple dots
        dots = []
        for r in results:
            color = "#ccff33" if r == "W" else "#2c2c2c"
            dots.append(f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};margin-right:10px;'></span>")
        st.markdown("".join(dots), unsafe_allow_html=True)
    card_close()

    card_open("Mejor Racha")
    best = hist.best_streak(surface=(None if st.session_state.filter_surface == "Todas" else st.session_state.filter_surface))
    st.markdown(f"**{best}** victorias seguidas")
    card_close()

    card_open("Superficies")
    order = ["Tierra batida", "Pista rápida", "Hierba", "Indoor"]
    surf = agg["surfaces"]
    for srf in order:
        w = surf.get(srf, {}).get("w", 0)
        t_ = surf.get(srf, {}).get("t", 0)
        pct = (w / t_ * 100.0) if t_ else 0.0
        st.markdown(f"**{pct:.0f}%** · Victorias en **{srf}** — {w} de {t_}")
    card_close()

    st.button("⬅️ Volver a LIVE", use_container_width=True, on_click=lambda: go("live"))


# ==========================================================
# ROUTER
# ==========================================================
if st.session_state.page == "live":
    page_live()
elif st.session_state.page == "analysis":
    page_analysis()
else:
    page_stats()
