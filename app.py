import json
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, date, time
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

/* ======================================================
   FIX: Tabs (LIVE / Analysis / Stats) invisibles en móvil
   ====================================================== */
div[data-baseweb="tab-list"] {
  gap: 0.4rem !important;
  padding-bottom: 0.2rem !important;
}
div[data-baseweb="tab"] button {
  color: #111 !important;
  font-weight: 800 !important;
  font-size: 1.02rem !important;
  background: #f3f4f6 !important;
  border-radius: 999px !important;
  padding: 0.42rem 0.75rem !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
}
div[data-baseweb="tab"][aria-selected="true"] button {
  background: #111 !important;
  color: #fff !important;
  border-color: #111 !important;
}
div[data-baseweb="tab-highlight"] {
  background: transparent !important; /* evita subrayado raro */
}
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)


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
        pressure_won = sum(1 for p in self.points if p.get("pressure") and p["result"] == "win")

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
# CALENDARIO (EVENTOS)
# ==========================================================
class CalendarStore:
    def __init__(self):
        self.events = []  # cada evento: {id,title,date,time,location,notes,created_at}

    def add(self, e: dict):
        self.events.append(e)

    def delete(self, idx: int):
        if 0 <= idx < len(self.events):
            self.events.pop(idx)

    def sort_events(self):
        def keyf(ev):
            d = ev.get("date") or "9999-12-31"
            t = ev.get("time") or "23:59"
            return (d, t)
        self.events.sort(key=keyf)


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
    if "calendar" not in st.session_state:
        st.session_state.calendar = CalendarStore()


ss_init()
live: LiveMatch = st.session_state.live
history: MatchHistory = st.session_state.history
calendar: CalendarStore = st.session_state.calendar

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


# ==========================================================
# NAV (TABS)
# ==========================================================
tabs = st.tabs(["🎾 LIVE", "📈 Analysis", "📊 Stats", "🗓️ Calendario"])


# ==========================================================
# TAB 1: LIVE
# ==========================================================
with tabs[0]:
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
        st.button("📈 Ir a Analysis", use_container_width=True, disabled=True)
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
            surf_save = st.selectbox("Superficie (guardar)", SURFACES, index=SURFACES.index(live.surface))

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
                        "won_match": won_match,
                        "sets_w": int(sw), "sets_l": int(sl),
                        "games_w": int(gw), "games_l": int(gl),
                        "surface": surf_save,
                        **report,
                    })

                    live.surface = surf_save
                    live.reset()
                    st.session_state.finish = None
                    st.session_state._open_finish = False
                    st.success("Partido guardado ✅")
                    st.rerun()

    st.divider()

    st.subheader("Exportar", anchor=False)
    small_note("Aquí ves tu historial, puedes editarlo/borrarlo y exportarlo/importarlo en JSON.")

    if not history.matches:
        st.info("Aún no hay partidos guardados.")
    else:
        matches = list(reversed(history.matches))

        for idx, m in enumerate(matches):
            real_i = len(history.matches) - 1 - idx
            date_ = m.get("date", "")
            surf = m.get("surface", "—")
            res = "✅ W" if m.get("won_match") else "❌ L"
            score = f"{m.get('sets_w',0)}-{m.get('sets_l',0)} sets · {m.get('games_w',0)}-{m.get('games_l',0)} juegos"
            pts = f"{m.get('points_won',0)}/{m.get('points_total',0)} pts ({m.get('points_pct',0):.0f}%)"

            with st.expander(f"{res} · {score} · {surf} · {date_}", expanded=False):
                st.write(f"**{score}**")
                small_note(f"{pts} · Presión: {m.get('pressure_won',0)}/{m.get('pressure_total',0)} ({m.get('pressure_pct',0):.0f}%)")

                fin = (m.get("finishes") or {})
                fin_line = f"Winners {fin.get('winner',0)} · ENF {fin.get('unforced',0)} · EF {fin.get('forced',0)} · Ace {fin.get('ace',0)} · DF {fin.get('double_fault',0)}"
                small_note(fin_line)

                e1, e2 = st.columns(2, gap="small")
                with e1:
                    if st.button("✏️ Editar", key=f"edit_btn_{m.get('id',real_i)}", use_container_width=True):
                        st.session_state._edit_index = real_i
                        st.session_state._edit_open = True
                        st.rerun()
                with e2:
                    if st.button("🗑️ Borrar", key=f"del_btn_{m.get('id',real_i)}", use_container_width=True):
                        history.matches.pop(real_i)
                        st.success("Partido borrado.")
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

                    date_txt = st.text_input("Fecha (ISO)", value=str(m.get("date", "")))

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
                            m["date"] = date_txt
                            history.matches[i] = m
                            st.session_state._edit_open = False
                            st.session_state._edit_index = None
                            st.success("Cambios guardados ✅")
                            st.rerun()

    export_obj = {"matches": history.matches}
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Descargar historial (JSON)",
        data=export_json,
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True,
    )

    up = st.file_uploader("⬆️ Importar historial (JSON)", type=["json"], label_visibility="visible")
    if up is not None:
        try:
            obj = json.loads(up.read().decode("utf-8"))
            matches_in = obj.get("matches", [])
            if not isinstance(matches_in, list):
                raise ValueError("Formato incorrecto: 'matches' debe ser una lista.")
            for mm in matches_in:
                if "id" not in mm:
                    mm["id"] = f"m_{datetime.now().timestamp()}"
            history.matches = matches_in
            st.success("Historial importado ✅")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo importar: {e}")


# ==========================================================
# TAB 2: ANALYSIS
# ==========================================================
with tabs[1]:
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
# TAB 3: STATS
# ==========================================================
with tabs[2]:
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


# ==========================================================
# TAB 4: CALENDARIO
# ==========================================================
with tabs[3]:
    title_h("Calendario")

    small_note("Apunta tus eventos (entrenos, partidos, torneos, fisio, etc.).")

    with st.expander("➕ Añadir evento", expanded=True):
        c1, c2 = st.columns([1.1, 0.9], gap="small")
        with c1:
            ev_title = st.text_input("Título", placeholder="Ej: Entreno / Partido / Torneo", key="cal_title")
            ev_loc = st.text_input("Lugar (opcional)", placeholder="Ej: Pinter / Club / Ciudad", key="cal_loc")
        with c2:
            ev_date = st.date_input("Fecha", value=date.today(), key="cal_date")
            ev_time = st.time_input("Hora (opcional)", value=time(18, 0), key="cal_time")

        ev_notes = st.text_area("Notas (opcional)", placeholder="Objetivo, rival, pista, recordatorios…", key="cal_notes")

        if st.button("Guardar evento", use_container_width=True, key="cal_add"):
            eid = f"e_{datetime.now().timestamp()}"
            calendar.add({
                "id": eid,
                "title": (ev_title or "Evento").strip(),
                "date": ev_date.isoformat(),
                "time": ev_time.strftime("%H:%M") if isinstance(ev_time, time) else "",
                "location": (ev_loc or "").strip(),
                "notes": (ev_notes or "").strip(),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
            calendar.sort_events()
            st.success("Evento guardado ✅")
            st.rerun()

    st.divider()
    st.subheader("Próximos eventos", anchor=False)

    if not calendar.events:
        st.info("Aún no hay eventos.")
    else:
        calendar.sort_events()
        for idx, ev in enumerate(calendar.events):
            ev_title = ev.get("title", "Evento")
            ev_date = ev.get("date", "")
            ev_time = ev.get("time", "")
            ev_loc = ev.get("location", "")
            header = f"{ev_date} {ev_time}".strip()
            if ev_loc:
                header += f" · {ev_loc}"
            with st.expander(f"🗓️ {ev_title} · {header}", expanded=False):
                if ev_loc:
                    st.write(f"**Lugar:** {ev_loc}")
                st.write(f"**Fecha:** {ev_date}" + (f" · **Hora:** {ev_time}" if ev_time else ""))
                if ev.get("notes"):
                    small_note(ev["notes"])

                e1, e2 = st.columns(2, gap="small")
                with e1:
                    if st.button("✏️ Editar", use_container_width=True, key=f"cal_edit_{ev.get('id', idx)}"):
                        st.session_state._cal_edit_open = True
                        st.session_state._cal_edit_idx = idx
                        st.rerun()
                with e2:
                    if st.button("🗑️ Borrar", use_container_width=True, key=f"cal_del_{ev.get('id', idx)}"):
                        calendar.delete(idx)
                        st.success("Evento borrado ✅")
                        st.rerun()

        if st.session_state.get("_cal_edit_open", False):
            i = st.session_state.get("_cal_edit_idx", None)
            if i is not None and 0 <= i < len(calendar.events):
                ev = calendar.events[i]
                with st.expander("✏️ Editar evento", expanded=True):
                    t_ = st.text_input("Título", value=str(ev.get("title", "")), key="cal_edit_title")
                    loc_ = st.text_input("Lugar (opcional)", value=str(ev.get("location", "")), key="cal_edit_loc")

                    # parse date/time
                    try:
                        d0 = date.fromisoformat(ev.get("date", date.today().isoformat()))
                    except Exception:
                        d0 = date.today()
                    try:
                        tm = ev.get("time", "")
                        if tm:
                            hh, mm = tm.split(":")
                            t0 = time(int(hh), int(mm))
                        else:
                            t0 = time(18, 0)
                    except Exception:
                        t0 = time(18, 0)

                    d_ = st.date_input("Fecha", value=d0, key="cal_edit_date")
                    t_in = st.time_input("Hora (opcional)", value=t0, key="cal_edit_time")
                    notes_ = st.text_area("Notas (opcional)", value=str(ev.get("notes", "")), key="cal_edit_notes")

                    bL, bR = st.columns(2, gap="small")
                    with bL:
                        if st.button("Cancelar edición", use_container_width=True, key="cal_edit_cancel"):
                            st.session_state._cal_edit_open = False
                            st.session_state._cal_edit_idx = None
                            st.rerun()
                    with bR:
                        if st.button("Guardar cambios", use_container_width=True, key="cal_edit_save"):
                            ev["title"] = (t_ or "Evento").strip()
                            ev["location"] = (loc_ or "").strip()
                            ev["date"] = d_.isoformat()
                            ev["time"] = t_in.strftime("%H:%M") if isinstance(t_in, time) else ""
                            ev["notes"] = (notes_ or "").strip()
                            calendar.events[i] = ev
                            calendar.sort_events()
                            st.session_state._cal_edit_open = False
                            st.session_state._cal_edit_idx = None
                            st.success("Cambios guardados ✅")
                            st.rerun()

    st.divider()
    st.subheader("Exportar / Importar calendario", anchor=False)

    cal_obj = {"events": calendar.events}
    cal_json = json.dumps(cal_obj, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Descargar calendario (JSON)",
        data=cal_json,
        file_name="tennis_calendar.json",
        mime="application/json",
        use_container_width=True,
    )

    cal_up = st.file_uploader("⬆️ Importar calendario (JSON)", type=["json"], key="cal_uploader")
    if cal_up is not None:
        try:
            obj = json.loads(cal_up.read().decode("utf-8"))
            evs = obj.get("events", [])
            if not isinstance(evs, list):
                raise ValueError("Formato incorrecto: 'events' debe ser una lista.")
            for e in evs:
                if "id" not in e:
                    e["id"] = f"e_{datetime.now().timestamp()}"
                if "date" not in e:
                    e["date"] = date.today().isoformat()
                if "title" not in e:
                    e["title"] = "Evento"
            calendar.events = evs
            calendar.sort_events()
            st.success("Calendario importado ✅")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo importar: {e}")
