# app.py
from __future__ import annotations

import json
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="TennisStats", page_icon="🎾", layout="centered")

# ==========================================================
# ESTILO (compacto + título clicable)
# ==========================================================
st.markdown(
    """
<style>
/* Compacto general */
.block-container {padding-top: 0.6rem; padding-bottom: 2.2rem;}
div[data-testid="stVerticalBlock"] > div {gap: 0.55rem;}
hr {margin: 0.6rem 0 !important;}
h1,h2,h3 {margin: 0.15rem 0 0.25rem 0 !important;}
div[data-testid="stWidget"] {margin-bottom: 0.25rem;}
@media (max-width: 480px){
  .block-container {padding-left: 0.85rem; padding-right: 0.85rem;}
}

/* Tarjetas */
.ts-card{
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.06);
  background: white;
}
.ts-card h3{margin:0 0 6px 0 !important;}
.ts-muted{color: rgba(0,0,0,0.55);}

/* Header row */
.ts-header-row{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap: 10px;
}

/* BOTÓN que parece un título */
button[kind="secondary"].ts-title-btn,
button[kind="primary"].ts-title-btn{
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
  margin: 0 !important;
  box-shadow: none !important;
  text-align: left !important;
}
button.ts-title-btn > div{
  font-size: 44px !important;
  font-weight: 800 !important;
  line-height: 1.02 !important;
  padding: 0 !important;
}
@media (max-width: 480px){
  button.ts-title-btn > div{font-size: 36px !important;}
}

/* Pequeño hint debajo */
.ts-nav-hint{
  font-size: 12px;
  color: rgba(0,0,0,0.55);
  margin-top: -6px;
}

/* Popover: botones full width */
.ts-nav-btn button{
  width: 100% !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# LÓGICA TENIS (Markov)
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

    def after_game(next_g_me: int, next_g_opp: int) -> float:
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
# MODELO LIVE + HISTORIAL
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
        self.points: List[Dict[str, Any]] = []
        self.state = LiveState()
        self.surface = "Tierra batida"
        self._undo: List[Tuple[LiveState, int, str]] = []

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

    def points_stats(self) -> Tuple[int, int, float]:
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

    def win_prob_series(self) -> List[float]:
        probs: List[float] = []
        tmp = LiveMatch()
        tmp.surface = self.surface
        for pt in self.points:
            tmp.add_point(pt["result"], {"finish": pt.get("finish")})
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
                "before": before.__dict__.copy(),
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

    def match_summary(self) -> Dict[str, Any]:
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

    def add(self, m: Dict[str, Any]):
        self.matches.append(m)

    def update(self, idx: int, m: Dict[str, Any]):
        if 0 <= idx < len(self.matches):
            self.matches[idx] = m

    def delete(self, idx: int):
        if 0 <= idx < len(self.matches):
            del self.matches[idx]

    def filtered_matches(self, n: Optional[int] = None, surface: Optional[str] = None):
        arr = self.matches[:]
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
            srf = m.get("surface", "—")
            surfaces.setdefault(srf, {"w": 0, "t": 0})
            surfaces[srf]["t"] += 1
            if m.get("won_match"):
                surfaces[srf]["w"] += 1

        points_total = sum(int(m.get("points_total", 0)) for m in matches)
        points_won = sum(int(m.get("points_won", 0)) for m in matches)
        pressure_total = sum(int(m.get("pressure_total", 0)) for m in matches)
        pressure_won = sum(int(m.get("pressure_won", 0)) for m in matches)

        finishes_sum = {"winner": 0, "unforced": 0, "forced": 0, "ace": 0, "double_fault": 0, "opp_error": 0, "opp_winner": 0}
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
# SESSION STATE
# ==========================================================
def _ensure_state():
    if "live" not in st.session_state:
        st.session_state.live = LiveMatch()
    if "history" not in st.session_state:
        st.session_state.history = MatchHistory()
    if "finish_selected" not in st.session_state:
        st.session_state.finish_selected = None
    if "nav" not in st.session_state:
        st.session_state.nav = "LIVE"
    if "history_selected_idx" not in st.session_state:
        st.session_state.history_selected_idx = None


_ensure_state()
live: LiveMatch = st.session_state.live
history: MatchHistory = st.session_state.history

# ==========================================================
# HELPERS UI
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


def card_open(title: str, subtitle: Optional[str] = None):
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="ts-muted">{subtitle}</div>', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def fmt_match_line(m: Dict[str, Any]) -> str:
    dt = m.get("date", "")
    try:
        dt = dt.replace("T", " ")
    except Exception:
        pass
    w = "✅ W" if m.get("won_match") else "❌ L"
    srf = m.get("surface", "—")
    return f"{w} · {m.get('sets_w',0)}-{m.get('sets_l',0)} sets · {m.get('games_w',0)}-{m.get('games_l',0)} juegos · {srf} · {dt}"


NAV_MAP = {
    "LIVE": "🎾 LIVE",
    "ANALYSIS": "📊 ANALYSIS",
    "STATS": "📈 STATS",
}


def set_nav(new_nav: str):
    if new_nav != st.session_state.nav:
        st.session_state.nav = new_nav
        st.rerun()


def header_click_title(title_text: str):
    """
    Lo que quieres:
    - Se ve el texto grande (LIVE MATCH / ANALYSIS / STATS)
    - Al clicar el texto, se abre un menú para cambiar de página.
    """
    left, right = st.columns([1.2, 0.8], vertical_alignment="bottom")

    with left:
        # Botón que parece título
        clicked = st.button(title_text, key=f"title_btn_{title_text}", type="secondary")
        # Aplicamos clase CSS al botón recién creado
        st.markdown(
            """
            <script>
            const btns = window.parent.document.querySelectorAll('button[kind="secondary"]');
            if(btns.length){
              const last = btns[btns.length-1];
              last.classList.add('ts-title-btn');
            }
            </script>
            """,
            unsafe_allow_html=True,
        )

        # Si el usuario clica el título, abrimos el popover usando session flag
        if clicked:
            st.session_state["_nav_popover_open"] = True

    with right:
        # Popover “menú” (aparece al lado, pero el gesto es: clic en el título -> aparece)
        open_now = bool(st.session_state.get("_nav_popover_open", False))
        with st.popover("Cambiar", use_container_width=True):
            st.markdown('<div class="ts-nav-btn">', unsafe_allow_html=True)
            st.caption("Ir a…")
            if st.button(NAV_MAP["LIVE"], use_container_width=True):
                st.session_state["_nav_popover_open"] = False
                set_nav("LIVE")
            if st.button(NAV_MAP["ANALYSIS"], use_container_width=True):
                st.session_state["_nav_popover_open"] = False
                set_nav("ANALYSIS")
            if st.button(NAV_MAP["STATS"], use_container_width=True):
                st.session_state["_nav_popover_open"] = False
                set_nav("STATS")
            st.markdown("</div>", unsafe_allow_html=True)

        # Si no estaba abierto por click, no hacemos nada (el popover se abre al tocar "Cambiar")
        # PERO: para el caso "click en título", forzamos un pequeño hint visual
        if open_now:
            st.markdown('<div class="ts-nav-hint">Menú abierto ✅</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ts-nav-hint">Toca el título para cambiar</div>', unsafe_allow_html=True)


# ==========================================================
# PANTALLAS
# ==========================================================
def screen_export_block():
    st.header("Exportar")
    st.caption("Aquí ves tu historial: puedes **editarlo / borrarlo** y **exportarlo / importarlo** en JSON.")

    if not history.matches:
        st.info("Aún no hay partidos guardados.")
    else:
        options = list(range(len(history.matches)))
        labels = [fmt_match_line(history.matches[i]) for i in options]

        idx = st.selectbox(
            "Historial (selecciona un partido)",
            options=options,
            format_func=lambda i: labels[i],
            index=options[-1] if st.session_state.history_selected_idx is None else st.session_state.history_selected_idx,
        )
        st.session_state.history_selected_idx = idx
        m = deepcopy(history.matches[idx])

        with st.expander("✏️ Editar partido seleccionado", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                m["won_match"] = st.checkbox("¿Victoria?", value=bool(m.get("won_match")))
                m["sets_w"] = int(st.number_input("Sets Yo", min_value=0, step=1, value=int(m.get("sets_w", 0))))
                m["games_w"] = int(st.number_input("Juegos Yo", min_value=0, step=1, value=int(m.get("games_w", 0))))
            with c2:
                m["sets_l"] = int(st.number_input("Sets Rival", min_value=0, step=1, value=int(m.get("sets_l", 0))))
                m["games_l"] = int(st.number_input("Juegos Rival", min_value=0, step=1, value=int(m.get("games_l", 0))))
                m["surface"] = st.selectbox(
                    "Superficie",
                    ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
                    index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(m.get("surface", "Tierra batida")),
                )

            auto = st.checkbox("Auto: victoria si sets yo > rival", value=True)
            if auto:
                m["won_match"] = (int(m.get("sets_w", 0)) > int(m.get("sets_l", 0)))

            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Guardar cambios", use_container_width=True):
                    history.update(idx, m)
                    st.success("Cambios guardados ✅")
                    st.rerun()
            with b2:
                if st.button("🗑️ Borrar partido", use_container_width=True):
                    history.delete(idx)
                    st.session_state.history_selected_idx = None
                    st.success("Partido borrado ✅")
                    st.rerun()

        with st.expander("📄 Detalle (solo lectura)", expanded=False):
            st.json(history.matches[idx])

    export_payload = {"matches": history.matches}
    export_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Descargar historial (JSON)",
        data=export_bytes,
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True,
    )

    up = st.file_uploader("Importar historial (JSON)", type=["json"])
    if up is not None:
        try:
            payload = json.loads(up.read().decode("utf-8"))
            new_matches = payload.get("matches", [])
            if isinstance(new_matches, list):
                history.matches = new_matches
                st.success(f"Importado ✅ ({len(new_matches)} partidos)")
                st.rerun()
            else:
                st.error("El JSON no tiene el formato correcto (se esperaba clave 'matches' con una lista).")
        except Exception as e:
            st.error(f"No se pudo leer el JSON: {e}")


def screen_live():
    header_click_title("LIVE MATCH")

    total, won, pct = live.points_stats()

    colA, colB = st.columns([1.25, 1])
    with colA:
        live.surface = st.selectbox(
            "Superficie",
            ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
            index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(live.surface),
        )
    with colB:
        st.markdown(f"**Puntos:** {total} · **% ganados:** {pct:.1f}%", unsafe_allow_html=False)

    st.divider()

    st.header("Marcador")
    st_ = live.state
    pts_label = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)
    st.write(f"**Sets {st_.sets_me}-{st_.sets_opp} · Juegos {st_.games_me}-{st_.games_opp} · Puntos {pts_label}**")

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0
    st.caption(f"Modelo: p(punto)≈{p_point:.2f}  ·  Win Prob≈{p_match:.1f}%")

    st.divider()

    st.header("Punto")

    with st.expander("Finish (opcional)", expanded=False):
        st.caption("Selecciona 1 (se aplica al siguiente punto). Puedes deseleccionar tocando de nuevo.")
        cols = st.columns(2)
        for i, (k, label) in enumerate(FINISH_ITEMS):
            with cols[i % 2]:
                is_on = (st.session_state.finish_selected == k)
                if st.button(("✅ " if is_on else "") + label, key=f"fin_{k}", use_container_width=True):
                    st.session_state.finish_selected = None if is_on else k
                    st.rerun()
        if st.button("Limpiar finish", use_container_width=False):
            st.session_state.finish_selected = None
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🟩 Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish_selected})
            st.session_state.finish_selected = None
            st.rerun()
    with c2:
        if st.button("🟥 Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish_selected})
            st.session_state.finish_selected = None
            st.rerun()

    with st.expander("Acciones manuales (+Juego / +Set)", expanded=False):
        g1, g2 = st.columns(2)
        with g1:
            if st.button("➕ Juego Yo", use_container_width=True):
                live.add_game_manual("me")
                st.rerun()
            if st.button("➕ Set Yo", use_container_width=True):
                live.add_set_manual("me")
                st.rerun()
        with g2:
            if st.button("➕ Juego Rival", use_container_width=True):
                live.add_game_manual("opp")
                st.rerun()
            if st.button("➕ Set Rival", use_container_width=True):
                live.add_set_manual("opp")
                st.rerun()

    st.divider()

    st.header("Acciones")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("↩️ Deshacer", use_container_width=True):
            live.undo()
            st.rerun()
    with a2:
        if st.button("🏁 Finalizar", use_container_width=True):
            st.session_state["_open_finish_modal"] = True
            st.rerun()
    with a3:
        if st.button("🧹 Reset live", use_container_width=True):
            live.reset()
            st.rerun()

    if st.session_state.get("_open_finish_modal"):
        card_open("Finalizar partido", "Guarda el resultado en el historial.")
        sw = st.number_input("Sets Yo", min_value=0, step=1, value=int(live.state.sets_me))
        sl = st.number_input("Sets Rival", min_value=0, step=1, value=int(live.state.sets_opp))
        gw = st.number_input("Juegos Yo", min_value=0, step=1, value=int(live.state.games_me))
        gl = st.number_input("Juegos Rival", min_value=0, step=1, value=int(live.state.games_opp))
        srf = st.selectbox(
            "Superficie (guardar)",
            ["Tierra batida", "Pista rápida", "Hierba", "Indoor"],
            index=["Tierra batida", "Pista rápida", "Hierba", "Indoor"].index(live.surface),
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Cancelar", use_container_width=True):
                st.session_state["_open_finish_modal"] = False
                st.rerun()
        with b2:
            if st.button("Guardar partido", use_container_width=True):
                won_match = sw > sl
                report = live.match_summary()
                history.add(
                    {
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "won_match": won_match,
                        "sets_w": int(sw),
                        "sets_l": int(sl),
                        "games_w": int(gw),
                        "games_l": int(gl),
                        "surface": srf,
                        **report,
                    }
                )
                live.surface = srf
                live.reset()
                st.session_state["_open_finish_modal"] = False
                st.success("Partido guardado ✅")
                st.rerun()
        card_close()

    st.divider()
    screen_export_block()


def screen_analysis():
    header_click_title("ANALYSIS")

    p_point = live.estimate_point_win_prob()
    p_match = live.match_win_prob() * 100.0

    card_open("Win Probability (modelo real)")
    st.write(f"**p(punto)≈{p_point:.2f} · Win Prob≈{p_match:.1f}%**")
    st.caption("Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.")

    probs = live.win_prob_series()
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        st.line_chart(probs, height=260)
    card_close()

    total = sum(1 for p in live.points if p.get("pressure"))
    won = sum(1 for p in live.points if p.get("pressure") and p["result"] == "win")
    pct = (won / total * 100.0) if total else 0.0

    card_open("Puntos de presión (live)")
    st.write(f"**{won}/{total}** ganados (**{pct:.0f}%**) en deuce/tiebreak.")
    card_close()


def screen_stats():
    header_click_title("STATS")

    card_open("Filtros")
    c1, c2 = st.columns([1, 1])
    with c1:
        n = st.selectbox("Partidos", ["Últ. 10", "Últ. 30", "Todos"], index=0)
        filter_n = 10 if n == "Últ. 10" else 30 if n == "Últ. 30" else None
    with c2:
        filter_surface = st.selectbox("Superficie", ["Todas", "Tierra batida", "Pista rápida", "Hierba", "Indoor"], index=0)
    card_close()

    agg = history.aggregate(n=filter_n, surface=filter_surface)

    card_open("Resumen")
    st.write(
        f"**Partidos:** {agg['matches_win']}/{agg['matches_total']} ({agg['matches_pct']:.0f}%)  ·  "
        f"**Sets:** {agg['sets_w']}/{agg['sets_w'] + agg['sets_l']} ({agg['sets_pct']:.0f}%)  ·  "
        f"**Juegos:** {agg['games_w']}/{agg['games_w'] + agg['games_l']} ({agg['games_pct']:.0f}%)"
    )
    st.caption(
        f"Puntos: {agg['points_won']}/{agg['points_total']} ({agg['points_pct']:.0f}%)  ·  "
        f"Presión: {agg['pressure_won']}/{agg['pressure_total']} ({agg['pressure_pct']:.0f}%)"
    )
    fin = agg["finishes_sum"]
    st.caption(
        f"Winners {fin['winner']} · ENF {fin['unforced']} · EF {fin['forced']} · "
        f"Aces {fin['ace']} · Dobles faltas {fin['double_fault']}"
    )
    card_close()

    card_open("Racha")
    results = history.last_n_results(10, surface=(None if filter_surface == "Todas" else filter_surface))
    if not results:
        st.info("Aún no hay partidos guardados.")
    else:
        st.write("Últimos 10: " + "  ".join(results))
    best = history.best_streak(surface=(None if filter_surface == "Todas" else filter_surface))
    st.write(f"Mejor racha: **{best}** victorias seguidas")
    card_close()

    order = ["Tierra batida", "Pista rápida", "Hierba", "Indoor"]
    card_open("Superficies")
    rows = []
    for srf in order:
        w = agg["surfaces"].get(srf, {}).get("w", 0)
        t = agg["surfaces"].get(srf, {}).get("t", 0)
        pct_ = (w / t * 100.0) if t else 0.0
        rows.append({"Superficie": srf, "Victorias": w, "Total": t, "%": round(pct_, 0)})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    card_close()


# ==========================================================
# ROUTER
# ==========================================================
if st.session_state.nav == "LIVE":
    screen_live()
elif st.session_state.nav == "ANALYSIS":
    screen_analysis()
else:
    screen_stats()
