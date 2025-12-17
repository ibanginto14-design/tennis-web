import json
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from functools import lru_cache

import pandas as pd
import streamlit as st


# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="TennisStats", layout="centered")


# ==========================================================
# ESTILO (móvil-friendly) + FIX títulos visibles
# ==========================================================
CSS = """
<style>
/* Fondo general */
.main { background: #f6f6f6; }

/* “Top bar” fake */
.appbar{
  position: sticky; top: 0; z-index: 999;
  margin: -1rem -1rem 1rem -1rem;
  padding: 0.9rem 1rem;
  background: #1f1f1f;
  border-bottom-left-radius: 18px;
  border-bottom-right-radius: 18px;
  box-shadow: 0 10px 20px rgba(0,0,0,.10);
}
.appbar h1{
  margin: 0;
  font-size: 1.15rem;
  color: #ffffff;
  letter-spacing: .5px;
  display: flex; align-items: center; gap: .5rem;
}
.appbar .sub{
  margin-top: .25rem;
  font-size: .85rem;
  color: rgba(255,255,255,.75);
}

/* Cards */
.card{
  background: #ffffff;
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 10px 20px rgba(0,0,0,.06);
  border: 1px solid rgba(0,0,0,.03);
  margin-bottom: 14px;
}
.card h3{
  margin: 0 0 10px 0;
  font-size: 1.0rem;
  color: #1f1f1f;
}

/* Botones */
.stButton button{
  width: 100%;
  border-radius: 14px !important;
  padding: 0.75rem 0.9rem !important;
  font-weight: 700;
}

/* Chips */
.chips{
  display:flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 8px;
}
.chip{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding: 10px 14px;
  border-radius: 999px;
  background: #eaeaea;
  color: #111;
  border: 1px solid rgba(0,0,0,.06);
  font-weight: 700;
  font-size: 0.95rem;
}
.chip-on{
  background: #ccff33;
  border-color: rgba(0,0,0,.10);
}
.small{
  color:#555; font-size: .9rem; margin-top: 6px;
}

/* Tabla */
div[data-testid="stDataFrame"] { background: #fff; border-radius: 14px; }

/* Separadores */
.hr{
  height: 10px;
  background: #1f1f1f;
  border-radius: 999px;
  margin: 14px 0 12px 0;
  opacity: .10;
}

/* Oculta el “footer” */
footer {visibility: hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================
# LÓGICA TENIS (del modelo original)
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
# MODELO DE DATOS
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
            "before": before.__dict__,
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

    def delete(self, idx: int):
        if 0 <= idx < len(self.matches):
            self.matches.pop(idx)

    def update(self, idx: int, new_m: dict):
        if 0 <= idx < len(self.matches):
            self.matches[idx] = new_m

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

        sets_w = sum(m.get("sets_w", 0) for m in matches)
        sets_l = sum(m.get("sets_l", 0) for m in matches)
        games_w = sum(m.get("games_w", 0) for m in matches)
        games_l = sum(m.get("games_l", 0) for m in matches)

        surfaces = {}
        for m in matches:
            srf = m.get("surface", "Tierra batida")
            surfaces.setdefault(srf, {"w": 0, "t": 0})
            surfaces[srf]["t"] += 1
            if m.get("won_match"):
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
# SESSION STATE INIT
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "live"

if "live" not in st.session_state:
    st.session_state.live = LiveMatch()

if "history" not in st.session_state:
    st.session_state.history = MatchHistory()

if "finish" not in st.session_state:
    st.session_state.finish = None


SURFACES = ("Tierra batida", "Pista rápida", "Hierba", "Indoor")


# ==========================================================
# UI HELPERS
# ==========================================================
def appbar(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="appbar">
          <h1>🎾 {title}</h1>
          <div class="sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def card_open(title: str):
    st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def hr():
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


def fmt_match_title(m: dict, idx: int):
    d = (m.get("date") or "").replace("T", " ")
    res = "✅ W" if m.get("won_match") else "❌ L"
    srf = m.get("surface", "")
    return f"#{idx+1}  {res}  ·  {m.get('sets_w',0)}-{m.get('sets_l',0)} sets  ·  {m.get('games_w',0)}-{m.get('games_l',0)} games  ·  {srf}  ·  {d}"


def history_df(matches: list[dict]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame(columns=["date", "result", "sets", "games", "surface", "points_pct", "pressure_pct"])
    rows = []
    for m in matches:
        rows.append({
            "date": (m.get("date") or "").replace("T", " "),
            "result": "W" if m.get("won_match") else "L",
            "sets": f"{m.get('sets_w',0)}-{m.get('sets_l',0)}",
            "games": f"{m.get('games_w',0)}-{m.get('games_l',0)}",
            "surface": m.get("surface", ""),
            "points_pct": round(float(m.get("points_pct", 0.0) or 0.0), 1),
            "pressure_pct": round(float(m.get("pressure_pct", 0.0) or 0.0), 1),
        })
    return pd.DataFrame(rows)


# ==========================================================
# LIVE PAGE
# ==========================================================
def page_live():
    live: LiveMatch = st.session_state.live

    total, won, pct = live.points_stats()
    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    appbar("LIVE MATCH", f"Superficie: {live.surface}  ·  Puntos: {total}  ·  % ganados: {pct:.1f}%  ·  WinProb≈{p_match:.1f}%")

    # Navegación superior (móvil: apilado)
    card_open("Navegación")
    live.surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(live.surface))
    if st.button("📊 Analysis"):
        st.session_state.page = "analysis"
        st.rerun()
    if st.button("📈 Stats"):
        st.session_state.page = "stats"
        st.rerun()
    card_close()

    # Marcador
    st_ = live.state
    pts = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)

    card_open("Marcador")
    st.markdown(
        f"""
        <div style="font-size:1.4rem;font-weight:900;color:#111;">
          Sets {st_.sets_me}-{st_.sets_opp} &nbsp;·&nbsp; Juegos {st_.games_me}-{st_.games_opp} &nbsp;·&nbsp; Puntos {pts}
        </div>
        <div class="small">
          Modelo: p(punto)≈{p_point:.2f} &nbsp;·&nbsp; Win Prob≈{p_match:.1f}%
        </div>
        """,
        unsafe_allow_html=True
    )
    card_close()

    # Punto
    card_open("Punto")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟩 Punto Yo"):
            live.add_point("win", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    with col2:
        if st.button("🟥 Punto Rival"):
            live.add_point("lose", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("+ Juego Yo"):
            live.add_game_manual("me"); st.rerun()
    with c2:
        if st.button("+ Juego Rival"):
            live.add_game_manual("opp"); st.rerun()
    with c3:
        if st.button("+ Set Yo"):
            live.add_set_manual("me"); st.rerun()
    with c4:
        if st.button("+ Set Rival"):
            live.add_set_manual("opp"); st.rerun()
    card_close()

    # Finish (opcional) — SOLO chips (sin duplicar)
    FINISH_ITEMS = [
        ("winner", "Winner"),
        ("unforced", "ENF"),
        ("forced", "EF"),
        ("ace", "Ace"),
        ("double_fault", "Doble falta"),
        ("opp_error", "Error rival"),
        ("opp_winner", "Winner rival"),
    ]

    card_open("Finish (opcional)")
    st.caption("Selecciona 1 (se aplica al siguiente punto). Puedes deseleccionar tocando de nuevo.")
    cols = st.columns(2)
    for i, (k, label) in enumerate(FINISH_ITEMS):
        with cols[i % 2]:
            selected = (st.session_state.finish == k)
            if st.button(("✅ " if selected else "") + label, key=f"chip_{k}"):
                st.session_state.finish = None if selected else k
                st.rerun()

    if st.button("Limpiar Finish"):
        st.session_state.finish = None
        st.rerun()
    card_close()

    # Acciones
    card_open("Acciones")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↩️ Deshacer"):
            live.undo(); st.session_state.finish = None; st.rerun()
    with c2:
        if st.button("📊 Ir a Analysis"):
            st.session_state.page = "analysis"; st.rerun()
    with c3:
        if st.button("🏁 Finalizar"):
            st.session_state.page = "finish"
            st.rerun()
    card_close()

    # Exportar (con historial)
    export_block()


# ==========================================================
# FINISH / GUARDAR PARTIDO
# ==========================================================
def page_finish():
    live: LiveMatch = st.session_state.live
    history: MatchHistory = st.session_state.history

    appbar("FINALIZAR", "Guardar el resultado del partido en el historial")

    card_open("Guardar partido")
    st_ = live.state
    sw = st.number_input("Sets Yo", min_value=0, max_value=5, value=int(st_.sets_me), step=1)
    sl = st.number_input("Sets Rival", min_value=0, max_value=5, value=int(st_.sets_opp), step=1)
    gw = st.number_input("Juegos Yo", min_value=0, max_value=50, value=int(st_.games_me), step=1)
    gl = st.number_input("Juegos Rival", min_value=0, max_value=50, value=int(st_.games_opp), step=1)
    srf = st.selectbox("Superficie (guardar)", SURFACES, index=SURFACES.index(live.surface))

    if st.button("Guardar partido"):
        won_match = (sw > sl)
        report = live.match_summary()
        history.add({
            "date": datetime.now().isoformat(timespec="seconds"),
            "won_match": won_match,
            "sets_w": int(sw), "sets_l": int(sl),
            "games_w": int(gw), "games_l": int(gl),
            "surface": srf,
            **report,
        })
        live.surface = srf
        live.reset()
        st.session_state.finish = None
        st.success("Partido guardado ✅")
        st.session_state.page = "live"
        st.rerun()

    if st.button("⬅️ Volver sin guardar"):
        st.session_state.page = "live"
        st.rerun()
    card_close()


# ==========================================================
# ANALYSIS PAGE
# ==========================================================
def page_analysis():
    live: LiveMatch = st.session_state.live
    total, won, pct = live.points_stats()
    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    appbar("Analysis", f"p(punto)≈{p_point:.2f}  ·  Win Prob≈{p_match:.1f}%  ·  Puntos: {total}")

    card_open("Win Probability (modelo real)")
    st.write(f"**p(punto)≈{p_point:.2f}**  ·  **Win Prob≈{p_match:.1f}%**")
    st.caption("Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.")
    card_close()

    card_open("Gráfica")
    probs = live.win_prob_series()
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        df = pd.DataFrame({"punto": list(range(1, len(probs) + 1)), "win_prob_%": probs})
        st.line_chart(df.set_index("punto"))
    card_close()

    card_open("Puntos de presión (live)")
    pressure_total = sum(1 for p in live.points if p.get("pressure"))
    pressure_won = sum(1 for p in live.points if p.get("pressure") and p["result"] == "win")
    pressure_pct = (pressure_won / pressure_total * 100.0) if pressure_total else 0.0
    st.write(f"**{pressure_won}/{pressure_total}** ganados (**{pressure_pct:.0f}%**) en deuce/tiebreak.")
    card_close()

    if st.button("⬅️ Volver a LIVE"):
        st.session_state.page = "live"
        st.rerun()

    export_block()


# ==========================================================
# STATS PAGE
# ==========================================================
def page_stats():
    history: MatchHistory = st.session_state.history

    appbar("Stats", "Resumen y métricas de tu historial")

    card_open("Filtros")
    n = st.radio("Ventana", options=["Últ. 10", "Últ. 30", "Todos"], horizontal=True)
    if n == "Últ. 10":
        filter_n = 10
    elif n == "Últ. 30":
        filter_n = 30
    else:
        filter_n = None

    filter_surface = st.selectbox("Superficie", ["Todas", *SURFACES], index=0)
    card_close()

    agg = history.aggregate(n=filter_n, surface=filter_surface)

    hr()

    card_open("Resumen")
    st.write(f"**Partidos ganados:** {agg['matches_win']} / {agg['matches_total']}  (**{agg['matches_pct']:.0f}%**)")
    st.write(f"**Sets ganados:** {agg['sets_w']} / {agg['sets_w'] + agg['sets_l']}  (**{agg['sets_pct']:.0f}%**)")
    st.write(f"**Juegos ganados:** {agg['games_w']} / {agg['games_w'] + agg['games_l']}  (**{agg['games_pct']:.0f}%**)")
    st.write(f"**Puntos:** {agg['points_won']} / {agg['points_total']}  (**{agg['points_pct']:.0f}%**)")
    st.write(f"**Presión:** {agg['pressure_won']} / {agg['pressure_total']}  (**{agg['pressure_pct']:.0f}%**)")
    card_close()

    card_open("Mejor racha")
    best = history.best_streak(surface=None if filter_surface == "Todas" else filter_surface)
    st.write(f"**{best}** victorias seguidas")
    card_close()

    if st.button("⬅️ Volver a LIVE"):
        st.session_state.page = "live"
        st.rerun()

    export_block()


# ==========================================================
# EXPORT + HISTORIAL + EDITAR
# ==========================================================
def export_block():
    history: MatchHistory = st.session_state.history

    card_open("Exportar")
    st.caption("Aquí ves tu historial, puedes editarlo/borrarlo y exportarlo/importarlo en JSON.")

    # Historial (tabla)
    matches = history.matches
    if not matches:
        st.info("Aún no hay partidos guardados.")
    else:
        df = history_df(matches)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Descargar JSON
    payload = json.dumps(history.matches, ensure_ascii=False, indent=2)
    st.download_button(
        "Descargar historial (JSON)",
        data=payload.encode("utf-8"),
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True
    )

    # Importar JSON
    up = st.file_uploader("Importar historial (JSON)", type=["json"])
    if up is not None:
        try:
            imported = json.loads(up.read().decode("utf-8"))
            if isinstance(imported, list):
                # merge (append)
                for m in imported:
                    if isinstance(m, dict) and "date" in m and "surface" in m:
                        history.add(m)
                st.success("Historial importado ✅")
                st.rerun()
            else:
                st.error("El JSON debe ser una lista de partidos.")
        except Exception as e:
            st.error(f"No se pudo leer el JSON: {e}")

    hr()

    # ===== NUEVO: EDITAR PARTIDOS GUARDADOS =====
    card_open("Editar partidos guardados")
    if not matches:
        st.info("No hay partidos para editar todavía.")
        card_close()
        return

    # selector por índice (más fiable que por fecha)
    options = [fmt_match_title(m, i) for i, m in enumerate(matches)]
    sel = st.selectbox("Selecciona un partido", options=options)
    idx = options.index(sel)
    m = deepcopy(matches[idx])

    # Form de edición
    with st.form("edit_match_form", clear_on_submit=False):
        st.write("**Editar resultado**")
        c1, c2 = st.columns(2)
        with c1:
            sets_w = st.number_input("Sets Yo (guardado)", min_value=0, max_value=5, value=int(m.get("sets_w", 0)), step=1)
            games_w = st.number_input("Juegos Yo (guardado)", min_value=0, max_value=50, value=int(m.get("games_w", 0)), step=1)
        with c2:
            sets_l = st.number_input("Sets Rival (guardado)", min_value=0, max_value=5, value=int(m.get("sets_l", 0)), step=1)
            games_l = st.number_input("Juegos Rival (guardado)", min_value=0, max_value=50, value=int(m.get("games_l", 0)), step=1)

        surface = st.selectbox("Superficie", ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
                               index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(m.get("surface", "Tierra batida")))

        # Si quieres permitir ajustar la fecha (opcional):
        edit_date = st.checkbox("Editar fecha", value=False)
        date_str = m.get("date", datetime.now().isoformat(timespec="seconds"))
        new_date = date_str
        if edit_date:
            new_date = st.text_input("Fecha ISO (ej: 2025-12-17T15:40:00)", value=date_str)

        # recalcular won_match
        won_match = bool(int(sets_w) > int(sets_l))

        save = st.form_submit_button("Guardar cambios")
        delete = st.form_submit_button("🗑️ Borrar este partido")

    if save:
        # Actualiza manteniendo estadísticas extra (puntos/pressure/finishes) tal cual
        m["sets_w"] = int(sets_w)
        m["sets_l"] = int(sets_l)
        m["games_w"] = int(games_w)
        m["games_l"] = int(games_l)
        m["surface"] = surface
        m["won_match"] = won_match
        m["date"] = new_date

        history.update(idx, m)
        st.success("Cambios guardados ✅")
        st.rerun()

    if delete:
        history.delete(idx)
        st.success("Partido borrado ✅")
        st.rerun()

    card_close()
    card_close()


# ==========================================================
# ROUTER
# ==========================================================
page = st.session_state.page

if page == "live":
    page_live()
elif page == "analysis":
    page_analysis()
elif page == "stats":
    page_stats()
elif page == "finish":
    page_finish()
else:
    st.session_state.page = "live"
    st.rerun()
