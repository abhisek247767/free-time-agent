"""Step 11: measure router accuracy on labelled requests.

Only the tev1 decision is evaluated (no tools run), so this is fast and does
not log fake walks. Every decision is saved to data/router_log.db with
`correct` filled in, alongside real decisions judged in the CLI.

Run with:  uv run python scripts/eval_router.py
"""

import time

from free_time_agent.agent import accuracy, log_decision
from free_time_agent.agent.router import ROUTER_MODEL, LoopResult, decide

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
    ("Just finished a 4 km walk at KBR Park", "log_walk"),
    ("Did 3.5 km around Necklace Road this morning, log it", "log_walk"),
    ("Save my walk: 2 km in Lotus Park", "log_walk"),
    ("Write me a poem about cats", "other"),
    ("Explain quantum computing", "other"),
    ("What's 17 times 23?", "other"),
]


def main() -> None:
    correct = 0
    started = time.monotonic()
    for request, expected in LABELLED:
        d = decide(request)
        result = LoopResult(
            request=request,
            intent=d["choice"],
            confidence=d["probabilities"][d["choice"]],
            probabilities=d["probabilities"],
        )
        log_decision(result, expected=expected)
        ok = result.intent == expected
        correct += ok
        mark = "ok  " if ok else "MISS"
        print(f"{mark} {result.intent:9} ({result.confidence:.2f})  want {expected:9}  {request}")

    elapsed = time.monotonic() - started
    print(f"\n{ROUTER_MODEL}: {correct}/{len(LABELLED)} correct "
          f"({100 * correct / len(LABELLED):.0f}%), {elapsed / len(LABELLED):.1f}s per decision")
    print("All judged decisions so far:", accuracy(ROUTER_MODEL))


if __name__ == "__main__":
    main()
