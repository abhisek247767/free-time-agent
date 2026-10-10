"""Streamlit frontend for the free-time walk planner.

Run with:  uv run streamlit run streamlit_app.py
"""

import time

import pandas as pd
import streamlit as st

from free_time_agent.agent import DEFAULT_LOCATION, plan
from free_time_agent.agent.router import RESPONSE_MODEL, ROUTER_MODEL

EXAMPLES = [
    "I have 1 hour free near Salt Lake",
    "Will it rain near Victoria Memorial in the next few hours?",
    "What time is sunset at Princep Ghat today?",
]

st.set_page_config(page_title="Free Time Walk Planner", page_icon="🌳", layout="centered")

with st.sidebar:
    st.header("Settings")
    location = st.text_input("Your location", value=DEFAULT_LOCATION)
    st.caption(f"Router: `{ROUTER_MODEL}`  \nArguments & answer: `{RESPONSE_MODEL}`")

st.title("🌳 Free Time Walk Planner")
st.caption("Tell me how much free time you have. I'll pick a park, a leave-by time and what to carry.")

if "request" not in st.session_state:
    st.session_state.request = ""

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.request = example

with st.form("ask"):
    request = st.text_input("Your request", key="request", placeholder="e.g. I have 45 minutes free near Park Street")
    submitted = st.form_submit_button("Plan my walk", type="primary")

if submitted and request.strip():
    started = time.monotonic()
    with st.status("Thinking…", expanded=True) as status:
        try:
            result = plan(request.strip(), location.strip() or DEFAULT_LOCATION, on_step=st.write)
        except Exception as exc:
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(exc).__name__}: {exc}. Is Ollama running?")
            st.stop()
        status.update(label=f"Done in {time.monotonic() - started:.0f}s", state="complete", expanded=False)

    with st.container(border=True):
        st.markdown(f"### {result.headline}")
        for line in result.lines:
            st.markdown(f"<div style='font-size:1.15rem;margin:0.3rem 0'>{line}</div>", unsafe_allow_html=True)
    loop = result.loop

    # Map: the geocoded location plus nearby parks, when the walk tools ran.
    here = loop.output_of("geocode")
    parks = (loop.output_of("find_green_spaces") or {}).get("places", [])
    if here:
        points = [{"lat": here["lat"], "lon": here["lon"], "label": "You", "color": "#1f77b4"}]
        points += [{"lat": p["lat"], "lon": p["lon"], "label": p["name"], "color": "#2ca02c"} for p in parks]
        st.map(pd.DataFrame(points), latitude="lat", longitude="lon", color="color", size=40)
        if parks:
            st.caption("Blue: your location · Green: nearby parks")

    with st.expander("How this answer was made"):
        st.markdown(f"**Router action:** `{loop.intent}` ({loop.confidence:.0%} confident)")
        if result.templated:
            st.caption("The model's headline failed the checks, so a fixed headline was used.")
        for run in loop.runs:
            st.markdown(f"**Round {run.round} · `{run.name}`**")
            st.json({"args": run.args, "result": run.output}, expanded=False)
elif submitted:
    st.warning("Type a request first.")
