vimport json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =========================
# Config
# =========================
st.set_page_config(page_title="TennisStats", layout="centered")

SURFACES = ["Tierra batida", "Pista rápida", "Hierba", "Indoor"]
FINISH_TYPES = ["Winner", "ENF", "EF", "Ace", "Doble falta", "Error rival", "Winner rival"]

HISTORY_FILE = "history.json"


# =========================
# Estilos
# =========================
st.markdown(
    """
<style>
.block-container {padding-top: 1.55rem; padding-bottom: 2.2rem;}
div[data-testid="stVerticalBlock"] > div {gap: 0.55rem;}
hr {margin: 0.6rem 0 !important;}
h1,h2,h3 {margin: 0.25rem 0 0.35rem 0 !important; line-height: 1.15 !important;}
div[data-testid="stWidget"] {margin-bottom: 0.25rem;}
@media (max-width: 480px){
  .block-container {padding-left: 0.85rem; padding-right: 0.85rem;}
}
.ts-card{
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.06);
  background: white;
}
.ts-muted{color: rgba(0,0,0,0.55);}
.ts-topbar{display:flex; align-items:center; justify-content:space-between; gap:12px; padding: 8px 2px 2px 2px;}
.ts-title{font-size: 30px; font-weight: 900; line-height: 1.1; margin:0; padding:0;}
.big-btn button{
  width: 100% !important;
  padding: 18px 16px !important;
  font-size: 18px !important;
  font-weight: 800 !important;
  border-radius: 16px !important;
}
.small-note{font-size: 12px; color: rgba(0,0,0,0.55);}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Datos
# =========================
@dataclass
class PointEvent:
    ts: str
    kind: str  # "pt:yo", "pt:rival", "finish:<tipo>", "manual:+juego_yo", etc.


@dataclass
class SavedMatch:
    id: str
    created_at: str
    surface: str
    sets_yo: int
    sets_rival: int
    games_yo: int
    games_rival: int
    tb_yo: int
    tb_rival: int
    in_tiebreak: bool
    notes: str
    events: List[Dict]


# =========================
# Persistencia de historial
# =========================
def load_history_from_disk() -> List[Dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history_to_disk(history: List[Dict]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        # si falla, al menos no rompemos la app
        pass


# =========================
# Estado
# =========================
def _init_state():
    ss = st.session_state
    ss.setdefault("page", "HOME")

    # Live
    ss.setdefault("surface_live", SURFACES[0])

    # Marcador tenis (game points y tiebreak)
    ss.setdefault("sets_yo", 0)
    ss.setdefault("sets_rival", 0)
    ss.setdefault("games_yo", 0)
    ss.setdefault("games_rival", 0)

    ss.setdefault("in_tiebreak", False)
    ss.setdefault("tb_yo", 0)
    ss.setdefault("tb_rival", 0)

    # puntos del juego (0,1,2,3,...)
    ss.setdefault("gp_yo", 0)
    ss.setdefault("gp_rival", 0)

    ss.setdefault("finish_selected", None)
    ss.setdefault("events", [])
    ss.setdefault("snapshots", [])  # para undo fiable

    # Historial persistente
    if "history" not in ss:
        ss.history = load_history_from_disk()

    # Stats filters
    ss.setdefault("stats_filter_n", "Todos")
    ss.setdefault("stats_filter_surface", "Todas")


_init_state()


def go(page: str):
    st.session_state.page = page
    st.rerun()


def top_bar(title: str):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f"""
            <div class="ts-topbar">
              <div class="ts-title">{title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("⬅️ Inicio", use_container_width=True):
            go("HOME")


# =========================
# Tennis scoring helpers
# =========================
def push_snapshot():
    ss = st.session_state
    snap = {
        "sets_yo": ss.sets_yo,
        "sets_rival": ss.sets_rival,
        "games_yo": ss.games_yo,
        "games_rival": ss.games_rival,
        "in_tiebreak": ss.in_tiebreak,
        "tb_yo": ss.tb_yo,
        "tb_rival": ss.tb_rival,
        "gp_yo": ss.gp_yo,
        "gp_rival": ss.gp_rival,
        "finish_selected": ss.finish_selected,
        "events_len": len(ss.events),
    }
    ss.snapshots.append(snap)


def undo_last():
    ss = st.session_state
    if not ss.snapshots:
        return
    snap = ss.snapshots.pop()

    ss.sets_yo = snap["sets_yo"]
    ss.sets_rival = snap["sets_rival"]
    ss.games_yo = snap["games_yo"]
    ss.games_rival = snap["games_rival"]
    ss.in_tiebreak = snap["in_tiebreak"]
    ss.tb_yo = snap["tb_yo"]
    ss.tb_rival = snap["tb_rival"]
    ss.gp_yo = snap["gp_yo"]
    ss.gp_rival = snap["gp_rival"]
    ss.finish_selected = snap["finish_selected"]

    # recorta eventos al tamaño que había antes de la acción
    ss.events = ss.events[: snap["events_len"]]


def add_event(kind: str):
    st.session_state.events.append(
        asdict(PointEvent(ts=datetime.now().isoformat(timespec="seconds"), kind=kind))
    )


def reset_live():
    ss = st.session_state
    ss.sets_yo = 0
    ss.sets_rival = 0
    ss.games_yo = 0
    ss.games_rival = 0
    ss.in_tiebreak = False
    ss.tb_yo = 0
    ss.tb_rival = 0
    ss.gp_yo = 0
    ss.gp_rival = 0
    ss.finish_selected = None
    ss.events = []
    ss.snapshots = []


def maybe_start_tiebreak():
    ss = st.session_state
    if ss.games_yo == 6 and ss.games_rival == 6:
        ss.in_tiebreak = True
        ss.tb_yo = 0
        ss.tb_rival = 0
        ss.gp_yo = 0
        ss.gp_rival = 0


def win_set(winner: str):
    ss = st.session_state
    if winner == "yo":
        ss.sets_yo += 1
    else:
        ss.sets_rival += 1

    # resetea juegos y tiebreak para nuevo set
    ss.games_yo = 0
    ss.games_rival = 0
    ss.in_tiebreak = False
    ss.tb_yo = 0
    ss.tb_rival = 0
    ss.gp_yo = 0
    ss.gp_rival = 0


def check_set_end_normal():
    ss = st.session_state
    # set normal: >=6 y diferencia >=2
    if ss.games_yo >= 6 and (ss.games_yo - ss.games_rival) >= 2:
        win_set("yo")
    elif ss.games_rival >= 6 and (ss.games_rival - ss.games_yo) >= 2:
        win_set("rival")
    else:
        maybe_start_tiebreak()


def win_game(winner: str):
    ss = st.session_state
    if winner == "yo":
        ss.games_yo += 1
    else:
        ss.games_rival += 1

    # reset puntos del juego
    ss.gp_yo = 0
    ss.gp_rival = 0

    # tras ganar juego, comprobar si termina set o tiebreak
    check_set_end_normal()


def apply_point_normal(winner: str):
    ss = st.session_state
    if winner == "yo":
        ss.gp_yo += 1
    else:
        ss.gp_rival += 1

    # gana el juego si >=4 y diferencia >=2
    if ss.gp_yo >= 4 and (ss.gp_yo - ss.gp_rival) >= 2:
        win_game("yo")
    elif ss.gp_rival >= 4 and (ss.gp_rival - ss.gp_yo) >= 2:
        win_game("rival")


def check_tiebreak_end():
    ss = st.session_state
    # tiebreak: >=7 y diferencia >=2
    if ss.tb_yo >= 7 and (ss.tb_yo - ss.tb_rival) >= 2:
        # gana set quien gana tiebreak (7-6)
        ss.games_yo = 7
        ss.games_rival = 6
        win_set("yo")
    elif ss.tb_rival >= 7 and (ss.tb_rival - ss.tb_yo) >= 2:
        ss.games_rival = 7
        ss.games_yo = 6
        win_set("rival")


def apply_point_tiebreak(winner: str):
    ss = st.session_state
    if winner == "yo":
        ss.tb_yo += 1
    else:
        ss.tb_rival += 1
    check_tiebreak_end()


def point_button(winner: str):
    """
    Acción principal: sumar punto (con scoring real), aplicar finish seleccionado, y registrar eventos.
    """
    push_snapshot()

    add_event(f"pt:{winner}")

    ss = st.session_state
    if ss.in_tiebreak:
        apply_point_tiebreak(winner)
    else:
        apply_point_normal(winner)

    # Finish (si había)
    if ss.finish_selected:
        add_event(f"finish:{ss.finish_selected}")
        ss.finish_selected = None


def tennis_point_label(gp_a: int, gp_b: int) -> str:
    """
    Convierte puntos del juego (gp) a display 0/15/30/40/Deuce/Ad.
    Se usa por separado para cada jugador, pero devolvemos el string compuesto.
    """
    # Deuce / Ad
    if gp_a >= 3 and gp_b >= 3:
        if gp_a == gp_b:
            return "Deuce"
        if gp_a == gp_b + 1:
            return "Ad Yo"
        if gp_b == gp_a + 1:
            return "Ad Rival"
        # si diferencia >=2 habría acabado el juego, pero por seguridad:
        return "—"

    map_pts = {0: "0", 1: "15", 2: "30", 3: "40"}
    return f"{map_pts.get(gp_a,'40')} - {map_pts.get(gp_b,'40')}"


# =========================
# Guardado de partido
# =========================
def save_match(notes: str = ""):
    ss = st.session_state
    mid = f"M{int(datetime.now().timestamp())}"

    m = SavedMatch(
        id=mid,
        created_at=datetime.now().isoformat(timespec="seconds"),
        surface=ss.surface_live,
        sets_yo=int(ss.sets_yo),
        sets_rival=int(ss.sets_rival),
        games_yo=int(ss.games_yo),
        games_rival=int(ss.games_rival),
        tb_yo=int(ss.tb_yo),
        tb_rival=int(ss.tb_rival),
        in_tiebreak=bool(ss.in_tiebreak),
        notes=notes or "",
        events=list(ss.events),
    )
    ss.history.insert(0, asdict(m))
    save_history_to_disk(ss.history)


def history_df(history: List[Dict]) -> pd.DataFrame:
    cols = ["Fecha", "Superficie", "Sets", "Juegos", "TB", "Notas", "ID"]
    if not history:
        return pd.DataFrame(columns=cols)

    rows = []
    for m in history:
        rows.append(
            {
                "Fecha": m.get("created_at", ""),
                "Superficie": m.get("surface", ""),
                "Sets": f'{m.get("sets_yo",0)}-{m.get("sets_rival",0)}',
                "Juegos": f'{m.get("games_yo",0)}-{m.get("games_rival",0)}',
                "TB": f'{m.get("tb_yo",0)}-{m.get("tb_rival",0)}' if m.get("in_tiebreak") else "",
                "Notas": (m.get("notes", "") or "")[:80],
                "ID": m.get("id", ""),
            }
        )
    return pd.DataFrame(rows, columns=cols)


# =========================
# Export / Edit / Delete / Import
# =========================
def export_block(show_intro: bool):
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Exportar</div>', unsafe_allow_html=True)
    if show_intro:
        st.markdown(
            '<div class="ts-muted">Aquí ves tu historial: puedes editarlo/borrarlo y exportarlo/importarlo en JSON.</div>',
            unsafe_allow_html=True,
        )

    hist = st.session_state.history

    if not hist:
        st.info("Aún no hay partidos guardados.")
    else:
        df_all = history_df(hist)
        st.dataframe(df_all.drop(columns=["ID"]), use_container_width=True, hide_index=True)

        ids = [m["id"] for m in hist]
        sel = st.selectbox("Selecciona partido para editar/borrar", ids, index=0)
        m = next((x for x in hist if x["id"] == sel), None)

        if m:
            with st.expander("Editar partido", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(m["surface"]), key=f"edit_surface_{sel}")
                    sets_yo = st.number_input("Sets Yo", 0, 10, int(m["sets_yo"]), key=f"edit_sets_yo_{sel}")
                    games_yo = st.number_input("Juegos Yo", 0, 50, int(m["games_yo"]), key=f"edit_games_yo_{sel}")
                with c2:
                    sets_rival = st.number_input("Sets Rival", 0, 10, int(m["sets_rival"]), key=f"edit_sets_rival_{sel}")
                    games_rival = st.number_input("Juegos Rival", 0, 50, int(m["games_rival"]), key=f"edit_games_rival_{sel}")

                notes = st.text_area("Notas", value=m.get("notes", ""), key=f"edit_notes_{sel}")

                csave, cdel = st.columns([2, 1])
                with csave:
                    if st.button("💾 Guardar cambios", use_container_width=True, key=f"save_edit_{sel}"):
                        m["surface"] = surface
                        m["sets_yo"] = int(sets_yo)
                        m["sets_rival"] = int(sets_rival)
                        m["games_yo"] = int(games_yo)
                        m["games_rival"] = int(games_rival)
                        m["notes"] = notes
                        save_history_to_disk(st.session_state.history)
                        st.success("Partido actualizado.")
                        st.rerun()
                with cdel:
                    if st.button("🗑️ Borrar", use_container_width=True, key=f"del_{sel}"):
                        st.session_state.history = [x for x in st.session_state.history if x["id"] != sel]
                        save_history_to_disk(st.session_state.history)
                        st.warning("Partido borrado.")
                        st.rerun()

    st.write("")

    payload = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Descargar historial (JSON)",
        data=payload.encode("utf-8"),
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown('<div class="small-note" style="margin-top:8px;">Importar historial (JSON)</div>', unsafe_allow_html=True)
    up = st.file_uploader(" ", type=["json"], label_visibility="collapsed")
    if up is not None:
        try:
            raw = up.read().decode("utf-8")
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                st.session_state.history = loaded
                save_history_to_disk(st.session_state.history)
                st.success("Historial importado correctamente.")
                st.rerun()
            else:
                st.error("El JSON no parece una lista de partidos.")
        except Exception as e:
            st.error(f"Error al importar JSON: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Páginas
# =========================
def page_home():
    top_bar("TennisStats")
    st.markdown(
        """
<div class="ts-card">
  <div style="font-size:16px;font-weight:700;margin-bottom:6px;">Elige una página</div>
  <div class="ts-muted">Empieza seleccionando qué quieres ver.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="big-btn">', unsafe_allow_html=True)
        if st.button("🎾 LIVE", use_container_width=True):
            go("LIVE")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="big-btn">', unsafe_allow_html=True)
        if st.button("📊 ANALYSIS", use_container_width=True):
            go("ANALYSIS")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="big-btn">', unsafe_allow_html=True)
        if st.button("📈 STATS", use_container_width=True):
            go("STATS")
        st.markdown("</div>", unsafe_allow_html=True)


def page_live():
    top_bar("LIVE MATCH")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.selectbox("Superficie", SURFACES, key="surface_live")

    with c2:
        ss = st.session_state
        # puntos totales del juego (solo para info rápida, no es marcador real)
        pts_total = (ss.gp_yo + ss.gp_rival) if not ss.in_tiebreak else (ss.tb_yo + ss.tb_rival)
        st.markdown(
            f"""
<div class="ts-card">
  <div style="font-weight:800;">Rallys: {pts_total}</div>
  <div class="ts-muted">Tiebreak: {"Sí" if ss.in_tiebreak else "No"}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()

    ss = st.session_state

    # Marcador
    if ss.in_tiebreak:
        point_display = f"TB {ss.tb_yo}-{ss.tb_rival}"
    else:
        point_display = tennis_point_label(ss.gp_yo, ss.gp_rival)

    st.markdown(
        f"""
<div class="ts-card">
  <div style="font-size:18px;font-weight:900;margin-bottom:6px;">Marcador</div>
  <div style="font-size:22px;font-weight:900;">
    Sets {ss.sets_yo}-{ss.sets_rival} &nbsp;•&nbsp;
    Juegos {ss.games_yo}-{ss.games_rival} &nbsp;•&nbsp;
    {point_display}
  </div>
  <div class="ts-muted" style="margin-top:6px;">
    {("Tiebreak activo (7 con 2 de diferencia)" if ss.in_tiebreak else "Juego normal (15/30/40/Ad)")}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # Punto
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Punto</div>', unsafe_allow_html=True)

    # Finish (una sola lista, no duplicada)
    with st.expander("Finish (opcional)", expanded=False):
        st.caption("Selecciona 1 (se aplica al siguiente punto). Puedes deseleccionar tocando de nuevo.")
        cols = st.columns(2)
        for i, ft in enumerate(FINISH_TYPES):
            with cols[i % 2]:
                selected = (ss.finish_selected == ft)
                label = f"✅ {ft}" if selected else ft
                if st.button(label, use_container_width=True, key=f"finish_{ft}"):
                    ss.finish_selected = None if selected else ft
                    st.rerun()

    cpy, cpr = st.columns(2)
    with cpy:
        if st.button("🟩 Punto Yo", use_container_width=True):
            point_button("yo")
            st.rerun()
    with cpr:
        if st.button("🟥 Punto Rival", use_container_width=True):
            point_button("rival")
            st.rerun()

    # Manual compacto
    with st.expander("Acciones manuales (+Juego / +Set)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Juego Yo", use_container_width=True):
                push_snapshot()
                ss.games_yo += 1
                ss.gp_yo = 0
                ss.gp_rival = 0
                add_event("manual:+juego_yo")
                check_set_end_normal()
                st.rerun()
            if st.button("➕ Set Yo", use_container_width=True):
                push_snapshot()
                add_event("manual:+set_yo")
                win_set("yo")
                st.rerun()
        with c2:
            if st.button("➕ Juego Rival", use_container_width=True):
                push_snapshot()
                ss.games_rival += 1
                ss.gp_yo = 0
                ss.gp_rival = 0
                add_event("manual:+juego_rival")
                check_set_end_normal()
                st.rerun()
            if st.button("➕ Set Rival", use_container_width=True):
                push_snapshot()
                add_event("manual:+set_rival")
                win_set("rival")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Acciones
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Acciones</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↩️ Deshacer", use_container_width=True):
            undo_last()
            st.rerun()
    with c2:
        with st.popover("🏁 Finalizar / Guardar"):
            notes = st.text_area("Notas (opcional)", key="notes_finish")
            if st.button("Guardar partido en historial", use_container_width=True):
                save_match(notes=notes)
                reset_live()
                st.success("Partido guardado ✅")
                st.rerun()
    with c3:
        if st.button("🧹 Reset live", use_container_width=True):
            reset_live()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    export_block(show_intro=True)


def page_analysis():
    top_bar("ANALYSIS")
    st.markdown(
        """
<div class="ts-card">
  <div style="font-size:18px;font-weight:900;margin-bottom:6px;">Análisis</div>
  <div class="ts-muted">Aquí puedes añadir análisis más avanzados cuando quieras.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_stats():
    top_bar("STATS")

    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Historial</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    hist = st.session_state.history
    df = history_df(hist)
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

    st.divider()
    export_block(show_intro=False)


# =========================
# Router
# =========================
page = st.session_state.page
if page == "HOME":
    page_home()
elif page == "LIVE":
    page_live()
elif page == "ANALYSIS":
    page_analysis()
elif page == "STATS":
    page_stats()
else:
    st.session_state.page = "HOME"
    st.rerun()
