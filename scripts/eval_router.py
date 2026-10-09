"""Check which action the router model picks for some labelled requests.

Only the router decision runs (no tools, no Qwen), so this is fast.
Nothing is stored; results are printed.

Run with:  uv run python scripts/eval_router.py
"""

from free_time_agent.agent.router import ROUTER_MODEL, decide

LABELLED = [
    ("I have 1 hour free near Gachibowli", "plan_walk"),
    ("Any park within walking distance of Madhapur?", "plan_walk"),
    ("Is there enough daylight left for a walk near Jubilee Hills?", "plan_walk"),
    ("Got 30 minutes before my next meeting, I'm in Hitech City", "plan_walk"),
    ("Suggest an evening stroll around Banjara Hills", "plan_walk"),
    ("Will it rain in Kondapur in the next few hours?", "weather"),
    ("How hot is it in Banjara Hills right now?", "weather"),
    ("Is it windy at Hussain Sagar?", "weather"),
    ("What time is sunset at Hussain Sagar today?", "daylight"),
    ("When is golden hour near Durgam Cheruvu?", "daylight"),
    ("What time does the sun rise tomorrow in Secunderabad?", "daylight"),
    ("Write me a poem about cats", "other"),
    ("Explain quantum computing", "other"),
    ("What's 17 times 23?", "other"),
]

correct = 0
for request, expected in LABELLED:
    d = decide(request)
    ok = d["choice"] == expected
    correct += ok
    print(f"{'ok  ' if ok else 'MISS'} {d['choice']:9} ({d['probabilities'][d['choice']]:.2f})  want {expected:9}  {request}")

print(f"\n{ROUTER_MODEL}: {correct}/{len(LABELLED)} correct ({100 * correct / len(LABELLED):.0f}%)")
