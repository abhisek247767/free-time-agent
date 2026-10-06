import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a walk for your free time.")
    parser.add_argument("request", nargs="*", help='e.g. "I have 1 hour free near Gachibowli"')
    parser.add_argument("-v", "--verbose", action="store_true", help="show the router's tool calls")
    parser.add_argument("--no-feedback", action="store_true", help="don't ask if the tool choice was right")
    parser.add_argument("--stats", action="store_true", help="print router accuracy and exit")
    args = parser.parse_args()

    from free_time_agent.agent import accuracy, mark_correct, plan

    if args.stats:
        print(accuracy())
        return

    request = " ".join(args.request) or input("What's your free time like? ")
    started = time.monotonic()
    result = plan(request)

    if args.verbose:
        loop = result.loop
        print(f"[router] action={loop.intent} confidence={loop.confidence:.2f}")
        for run in loop.runs:
            print(f"[round {run.round}] {run.name}({run.args})")
            print(f"    -> {run.output}")
        source = "template (model reply failed checks)" if result.templated else "model"
        print(f"[{result.loop.rounds} round(s), {time.monotonic() - started:.1f}s, answer from {source}]\n")

    print(result.answer)

    if not args.no_feedback and sys.stdin.isatty():
        loop = result.loop
        tools = ", ".join(loop.tools_called) or "none"
        answer = input(
            f"\nRouter chose '{loop.intent}' ({tools}). Right choice? [y/n, Enter to skip] "
        ).strip().lower()
        if answer in ("y", "n"):
            mark_correct(result.decision_id, answer == "y")
