"""Answer a request to the tool, as JSON (docs/plan-tool.md step 9).

    uv run python -m scripts.api --example > request.json     a complete example request
    uv run python -m scripts.api request.json                  its answer
    uv run python -m scripts.api - < request.json              the same, from standard input
    uv run python -m scripts.api --inputs                      every input, with the example firm's values

The request and the answer are described in `quant/api.py`. A request that
cannot be run is answered with `"ok": false` and the reasons, and the command
exits with status 1.
"""

import json
import sys

from quant.api import example_request, inputs, respond
from quant.cli import profile_parser
from quant.numerics import get_profile


def main(argv: list[str] | None = None) -> int:
    parser = profile_parser(__doc__.split("\n\n")[0])
    parser.add_argument("request", nargs="?", help="a JSON file, or - for standard input")
    parser.add_argument("--example", action="store_true", help="print an example request")
    parser.add_argument("--inputs", action="store_true", help="print every input a request can carry")
    args = parser.parse_args(argv)
    if args.example or args.inputs:
        print(json.dumps(example_request() if args.example else inputs(), indent=2))
        return 0
    if args.request is None:
        parser.error("give a request file, or - for standard input, or --example")
    try:
        text = sys.stdin.read() if args.request == "-" else open(args.request).read()
        request = json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        answer = {"ok": False, "problems": [f"the request could not be read: {error}"]}
    else:
        answer = respond(request, get_profile(args.profile))
    print(json.dumps(answer, indent=2))
    return 0 if answer["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
