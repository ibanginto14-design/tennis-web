import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# =========================
# Config
# =========================
st.set_page_config(page_title="TennisStats", layout="centered")


# =========================
# Estilos (incluye FIX de títulos)
# =========================
st.markdown(
    """
<style>
/* Más aire arriba para que no se recorte el primer header */
.block-container {padding-top: 1.55rem; padding-bottom: 2.2rem;}
div[data-testid="stVerticalBlock"] > div {gap: 0.55rem;}
hr {margin: 0.6rem 0 !important;}

/* Evita recortes por márgenes agresivos */
h1,h2,h3 {margin: 0.25rem 0 0.35rem 0 !important; line-height: 1.15 !important;}
div[data-testid="stWidget"] {margin-bottom: 0.25rem;}

@media (max-width: 480px){
  .block-container {padding-left: 0.85rem; padding-right: 0.85rem;}
}

/* Tarjetas / Cards */
.ts-card{
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.06);
  background: white;
}
.ts-muted{color: rgba(0,0,0,0.55);}

/* Botones grandes HOME */
.big-btn button{
  width: 100% !important;
  padding: 18px 16px !important;
  font-size: 18px !important;
  font-weight: 800 !important;
  border-radius: 16px !important;
}

/* Barra de título propia (no se recorta) */
.ts-topbar{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 2px 2px 2px;
}
.ts-title{
  font-size: 30px;
  font-weight: 900;
  line-height: 1.1;
  margin: 0;
  padding: 0;
}
.ts-right{
  display:flex;
  justify-content:flex-end;
  align-items:center;
  height:100%;
}
.small-note{
  font-size: 12px;
  color: rgba(0,0,0,0.55);
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Datos
# =========================
SURFACES = ["Tierra batida", "Pista rápida", "Hierba", "Indoor"]

FINISH_TYPES = [
    "Winner",
    "ENF",
    "EF",
    "Ace",
    "Doble falta",
    "Error rival",
    "Winner rival",
]


@dataclass
class PointEvent:
    ts: str
    kind: str  # "yo" / "rival" / "finish:<tipo>" / "manual:juego_yo" etc.


@dataclass
class SavedMatch:
    id: str
    created_at: str
    surface: str
    sets_yo: int
    sets_rival: int
    juegos_yo: int
    juegos_rival: int
    puntos_yo: int
    puntos_rival: int
    notes: str
    events: List[Dict]


# =========================
# Estado
# =========================
def _init_state():
    ss = st.session_state
    ss.setdefault("page", "HOME")

    # Live match state
    ss.setdefault("surface_live", SURFACES[0])
    ss.setdefault("sets_yo", 0)
    ss.setdefault("sets_rival", 0)
    ss.setdefault("juegos_yo", 0)
    ss.setdefault("juegos_rival", 0)
    ss.setdefault("puntos_yo", 0)
    ss.setdefault("puntos_rival", 0)
    ss.setdefault("finish_selected", None)  # uno solo (sin duplicados)
    ss.setdefault("events", [])  # list[PointEvent as dict]

    # History
    ss.setdefault("history", [])  # list[SavedMatch as dict]

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
        st.markdown('<div class="ts-right">', unsafe_allow_html=True)
        if st.button("⬅️ Inicio", use_container_width=True):
            go("HOME")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Lógica
# =========================
def add_event(kind: str):
    st.session_state.events.append(
        asdict(PointEvent(ts=datetime.now().isoformat(timespec="seconds"), kind=kind))
    )


def pct(n: int, d: int) -> float:
    return 0.0 if d <= 0 else (100.0 * n / d)


def p_point_estimate() -> float:
    """Estimación simple con tus puntos del partido (mínimo suavizado)."""
    yo = st.session_state.puntos_yo
    rv = st.session_state.puntos_rival
    total = yo + rv
    # suavizado para no explotar cuando hay pocos puntos
    return (yo + 1) / (total + 2)


def win_prob_from_p_point(p: float) -> float:
    """
    Aproximación simple (no exacta) para mostrar algo interpretable.
    Mantiene rango 0-1 y reacciona a cambios de p.
    """
    # curva sigmoide suave centrada en 0.5
    import math

    x = (p - 0.5) * 10.0
    return 1 / (1 + math.exp(-x))


def reset_live():
    ss = st.session_state
    ss.sets_yo = 0
    ss.sets_rival = 0
    ss.juegos_yo = 0
    ss.juegos_rival = 0
    ss.puntos_yo = 0
    ss.puntos_rival = 0
    ss.finish_selected = None
    ss.events = []


def undo_last():
    ss = st.session_state
    if not ss.events:
        return

    last = ss.events.pop()
    k = last["kind"]

    # revert kinds
    if k == "yo":
        ss.puntos_yo = max(0, ss.puntos_yo - 1)
    elif k == "rival":
        ss.puntos_rival = max(0, ss.puntos_rival - 1)
    elif k.startswith("manual:juego_yo"):
        ss.juegos_yo = max(0, ss.juegos_yo - 1)
    elif k.startswith("manual:juego_rival"):
        ss.juegos_rival = max(0, ss.juegos_rival - 1)
    elif k.startswith("manual:set_yo"):
        ss.sets_yo = max(0, ss.sets_yo - 1)
    elif k.startswith("manual:set_rival"):
        ss.sets_rival = max(0, ss.sets_rival - 1)
    elif k.startswith("finish:"):
        # sólo marca de finish (no cambia marcador)
        pass


def save_match(notes: str = ""):
    ss = st.session_state
    mid = f"M{int(datetime.now().timestamp())}"
    m = SavedMatch(
        id=mid,
        created_at=datetime.now().isoformat(timespec="seconds"),
        surface=ss.surface_live,
        sets_yo=int(ss.sets_yo),
        sets_rival=int(ss.sets_rival),
        juegos_yo=int(ss.juegos_yo),
        juegos_rival=int(ss.juegos_rival),
        puntos_yo=int(ss.puntos_yo),
        puntos_rival=int(ss.puntos_rival),
        notes=notes or "",
        events=list(ss.events),
    )
    ss.history.insert(0, asdict(m))


def history_df(history: List[Dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(
            columns=[
                "Fecha",
                "Superficie",
                "Sets",
                "Juegos",
                "Puntos",
                "Notas",
                "ID",
            ]
        )

    rows = []
    for m in history:
        rows.append(
            {
                "Fecha": m.get("created_at", ""),
                "Superficie": m.get("surface", ""),
                "Sets": f'{m.get("sets_yo",0)}-{m.get("sets_rival",0)}',
                "Juegos": f'{m.get("juegos_yo",0)}-{m.get("juegos_rival",0)}',
                "Puntos": f'{m.get("puntos_yo",0)}-{m.get("puntos_rival",0)}',
                "Notas": (m.get("notes", "") or "")[:80],
                "ID": m.get("id", ""),
            }
        )
    return pd.DataFrame(rows)


def filter_history() -> List[Dict]:
    ss = st.session_state
    hist = list(ss.history)

    # filtro superficie
    if ss.stats_filter_surface != "Todas":
        hist = [m for m in hist if m.get("surface") == ss.stats_filter_surface]

    # filtro n
    if ss.stats_filter_n == "Últ. 10":
        hist = hist[:10]
    elif ss.stats_filter_n == "Últ. 30":
        hist = hist[:30]

    return hist


# =========================
# UI: HOME
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


# =========================
# UI: LIVE
# =========================
def page_live():
    top_bar("LIVE MATCH")

    # Header compacto
    c1, c2 = st.columns([3, 2])
    with c1:
        st.selectbox("Superficie", SURFACES, key="surface_live")
    with c2:
        pts_total = st.session_state.puntos_yo + st.session_state.puntos_rival
        st.markdown(
            f"""
<div class="ts-card">
  <div style="font-weight:800;">Puntos: {pts_total} • {pct(st.session_state.puntos_yo, pts_total):.1f}% ganados</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()

    # Marcador
    p = p_point_estimate()
    wp = win_prob_from_p_point(p)
    st.markdown(
        f"""
<div class="ts-card">
  <div style="font-size:18px;font-weight:900;margin-bottom:6px;">Marcador</div>
  <div style="font-size:22px;font-weight:900;">
    Sets {st.session_state.sets_yo}-{st.session_state.sets_rival} &nbsp;•&nbsp;
    Juegos {st.session_state.juegos_yo}-{st.session_state.juegos_rival} &nbsp;•&nbsp;
    Puntos {st.session_state.puntos_yo}-{st.session_state.puntos_rival}
  </div>
  <div class="ts-muted" style="margin-top:6px;">
    Modelo: p(punto)≈{p:.2f} &nbsp;•&nbsp; Win Prob≈{100*wp:.1f}%
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # PUNTO: ahorrar espacio (finish plegable + manual plegable)
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Punto</div>', unsafe_allow_html=True)

    # Finish (opcional) — UNA SOLA LISTA (sin duplicar)
    with st.expander("Finish (opcional)", expanded=False):
        st.caption("Selecciona 1 (se aplica al siguiente punto). Puedes deseleccionar tocando de nuevo.")
        cols = st.columns(2)
        for i, ft in enumerate(FINISH_TYPES):
            with cols[i % 2]:
                selected = (st.session_state.finish_selected == ft)
                label = f"✅ {ft}" if selected else ft
                if st.button(label, use_container_width=True, key=f"finish_{ft}"):
                    # toggle
                    st.session_state.finish_selected = None if selected else ft
                    st.rerun()

    # Botones principales (punto yo / rival)
    cpy, cpr = st.columns(2)
    with cpy:
        if st.button("🟩 Punto Yo", use_container_width=True):
            st.session_state.puntos_yo += 1
            add_event("yo")
            if st.session_state.finish_selected:
                add_event(f"finish:{st.session_state.finish_selected}")
                st.session_state.finish_selected = None
            st.rerun()

    with cpr:
        if st.button("🟥 Punto Rival", use_container_width=True):
            st.session_state.puntos_rival += 1
            add_event("rival")
            if st.session_state.finish_selected:
                add_event(f"finish:{st.session_state.finish_selected}")
                st.session_state.finish_selected = None
            st.rerun()

    # Manual (plegable)
    with st.expander("Acciones manuales (+Juego / +Set)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Juego Yo", use_container_width=True):
                st.session_state.juegos_yo += 1
                add_event("manual:juego_yo")
                st.rerun()
            if st.button("➕ Set Yo", use_container_width=True):
                st.session_state.sets_yo += 1
                add_event("manual:set_yo")
                st.rerun()
        with c2:
            if st.button("➕ Juego Rival", use_container_width=True):
                st.session_state.juegos_rival += 1
                add_event("manual:juego_rival")
                st.rerun()
            if st.button("➕ Set Rival", use_container_width=True):
                st.session_state.sets_rival += 1
                add_event("manual:set_rival")
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
        if st.button("🏁 Finalizar", use_container_width=True):
            # Guardar y reset
            with st.popover("Guardar partido"):
                notes = st.text_area("Notas (opcional)", key="notes_finish")
                if st.button("Guardar en historial", use_container_width=True):
                    save_match(notes=notes)
                    reset_live()
                    st.success("Partido guardado.")
                    st.rerun()
    with c3:
        if st.button("🧹 Reset live", use_container_width=True):
            reset_live()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    export_block(show_intro=True)


# =========================
# UI: ANALYSIS
# =========================
def page_analysis():
    top_bar("ANALYSIS")

    p = p_point_estimate()
    wp = win_prob_from_p_point(p)

    st.markdown(
        f"""
<div class="ts-card">
  <div style="font-size:18px;font-weight:900;margin-bottom:6px;">Win Probability (modelo real)</div>
  <div style="font-size:18px;font-weight:800;">p(punto)≈{p:.2f} &nbsp;•&nbsp; Win Prob≈{100*wp:.1f}%</div>
  <div class="ts-muted" style="margin-top:6px;">
    Modelo: Markov (aprox). p(punto) se estima con tus puntos del partido.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Gráfico simple (si hay datos)
    pts_total = st.session_state.puntos_yo + st.session_state.puntos_rival
    if pts_total < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
        return

    # Serie de p_point por evento (aprox)
    yo = 0
    rv = 0
    series = []
    for e in st.session_state.events:
        if e["kind"] == "yo":
            yo += 1
        elif e["kind"] == "rival":
            rv += 1
        total = yo + rv
        if total > 0:
            series.append((total, (yo + 1) / (total + 2)))

    df = pd.DataFrame(series, columns=["Punto", "p_point_est"])
    st.line_chart(df.set_index("Punto"))

    st.divider()
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Puntos de presión (live)</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div>0/0 ganados (0%) en deuce/tiebreak (demo).</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# UI: STATS
# =========================
def page_stats():
    top_bar("STATS")

    # Filtros
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Filtros</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 2])
    with c1:
        st.session_state.stats_filter_n = st.radio(
            "Partidos",
            ["Últ. 10", "Últ. 30", "Todos"],
            horizontal=True,
            index=["Últ. 10", "Últ. 30", "Todos"].index(st.session_state.stats_filter_n),
        )
    with c2:
        st.session_state.stats_filter_surface = st.selectbox(
            "Superficie",
            ["Todas"] + SURFACES,
            index=(["Todas"] + SURFACES).index(st.session_state.stats_filter_surface),
        )
    st.markdown("</div>", unsafe_allow_html=True)

    hist = filter_history()

    # Resumen
    pts_w = sum(m.get("puntos_yo", 0) for m in hist)
    pts_l = sum(m.get("puntos_rival", 0) for m in hist)
    pts_total = pts_w + pts_l

    st.markdown(
        f"""
<div class="ts-card">
  <div style="font-size:18px;font-weight:900;margin-bottom:6px;">Resumen (filtro actual)</div>
  <div><b>Puntos:</b> {pts_w} ({pct(pts_w, pts_total):.1f}%) &nbsp;•&nbsp; <b>Presión:</b> 0/0 (0%)</div>
  <div class="ts-muted" style="margin-top:6px;">
    Winners: 0 • ENF: 0 • EF: 0 • Aces: 0 • Dobles faltas: 0
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Historial en tabla
    st.markdown(
        """
<div class="ts-card">
  <div style="font-size:18px;font-weight:900;margin-bottom:6px;">Historial</div>
</div>
""",
        unsafe_allow_html=True,
    )
    df = history_df(hist)
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

    st.divider()
    export_block(show_intro=False)


# =========================
# Exportar + Editar/Borrar + JSON import/export
# =========================
def export_block(show_intro: bool):
    st.markdown('<div class="ts-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:900;margin-bottom:6px;">Exportar</div>', unsafe_allow_html=True)

    if show_intro:
        st.markdown(
            '<div class="ts-muted">Aquí ves tu historial: puedes editarlo / borrarlo y exportarlo / importarlo en JSON.</div>',
            unsafe_allow_html=True,
        )

    hist = st.session_state.history

    # Historial visible
    if not hist:
        st.info("Aún no hay partidos guardados.")
    else:
        df_all = history_df(hist)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

        # Editar / borrar
        ids = [m["id"] for m in hist]
        sel = st.selectbox("Selecciona partido para editar/borrar", ids, index=0)
        m = next((x for x in hist if x["id"] == sel), None)

        if m:
            with st.expander("Editar partido", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(m["surface"]), key=f"edit_surface_{sel}")
                    sets_yo = st.number_input("Sets Yo", 0, 10, int(m["sets_yo"]), key=f"edit_sets_yo_{sel}")
                    juegos_yo = st.number_input("Juegos Yo", 0, 50, int(m["juegos_yo"]), key=f"edit_juegos_yo_{sel}")
                    puntos_yo = st.number_input("Puntos Yo", 0, 500, int(m["puntos_yo"]), key=f"edit_puntos_yo_{sel}")
                with c2:
                    sets_rival = st.number_input("Sets Rival", 0, 10, int(m["sets_rival"]), key=f"edit_sets_rival_{sel}")
                    juegos_rival = st.number_input("Juegos Rival", 0, 50, int(m["juegos_rival"]), key=f"edit_juegos_rival_{sel}")
                    puntos_rival = st.number_input("Puntos Rival", 0, 500, int(m["puntos_rival"]), key=f"edit_puntos_rival_{sel}")

                notes = st.text_area("Notas", value=m.get("notes", ""), key=f"edit_notes_{sel}")

                csave, cdel = st.columns([2, 1])
                with csave:
                    if st.button("💾 Guardar cambios", use_container_width=True, key=f"save_edit_{sel}"):
                        # aplicar cambios
                        m["surface"] = surface
                        m["sets_yo"] = int(sets_yo)
                        m["sets_rival"] = int(sets_rival)
                        m["juegos_yo"] = int(juegos_yo)
                        m["juegos_rival"] = int(juegos_rival)
                        m["puntos_yo"] = int(puntos_yo)
                        m["puntos_rival"] = int(puntos_rival)
                        m["notes"] = notes
                        st.success("Partido actualizado.")
                        st.rerun()
                with cdel:
                    if st.button("🗑️ Borrar", use_container_width=True, key=f"del_{sel}"):
                        st.session_state.history = [x for x in st.session_state.history if x["id"] != sel]
                        st.warning("Partido borrado.")
                        st.rerun()

    st.write("")

    # Descargar JSON
    payload = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Descargar historial (JSON)",
        data=payload.encode("utf-8"),
        file_name="tennis_history.json",
        mime="application/json",
        use_container_width=True,
    )

    # Importar JSON
    st.markdown('<div class="small-note" style="margin-top:8px;">Importar historial (JSON)</div>', unsafe_allow_html=True)
    up = st.file_uploader(" ", type=["json"], label_visibility="collapsed")
    if up is not None:
        try:
            raw = up.read().decode("utf-8")
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                # muy simple: reemplaza el historial
                st.session_state.history = loaded
                st.success("Historial importado correctamente.")
                st.rerun()
            else:
                st.error("El JSON no parece una lista de partidos.")
        except Exception as e:
            st.error(f"Error al importar JSON: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


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
    # fallback
    st.session_state.page = "HOME"
    st.rerun()
