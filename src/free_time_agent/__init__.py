import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a walk for your free time.")
    parser.add_argument("request", nargs="*", help='e.g. "I have 1 hour free near Gachibowli"')
    parser.add_argument("-v", "--verbose", action="store_true", help="show the router decision and tool calls")
    args = parser.parse_args()

    from free_time_agent.agent import plan

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
        print(f"[{loop.rounds} round(s), {time.monotonic() - started:.1f}s, answer from {source}]\n")

    print(result.answer)
