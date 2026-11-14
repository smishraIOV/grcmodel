"""The dashboard's server and page (docs/plan-tool.md step 10). The server
is the interface of `quant.api` over HTTP on this machine; the page is checked
only for what can break silently: that it is served, and that every element
its script looks for exists.
"""

import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from quant.api import plain
from quant.numerics import REFERENCE
from quant.profile import example_facts
from scripts.dashboard import PAGE, make_server


@pytest.fixture(scope="module")
def address():
    server = make_server(0, REFERENCE)                  # port 0: any free port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.headers["Content-Type"], response.read().decode()


def post(url, body: bytes):
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_server_listens_on_this_machine_only(address):
    assert address.startswith("http://127.0.0.1:")


def test_the_page_is_served_with_or_without_options(address):
    for path in ("/", "/index.html", "/?theme=light"):
        status, kind, body = get(address + path)
        assert status == 200 and kind.startswith("text/html") and "What-if: a firm under a policy" in body


def test_the_inputs_are_served(address):
    status, kind, body = get(address + "/inputs")
    listing = json.loads(body)
    assert status == 200 and kind.startswith("application/json")
    assert listing["firm"] == plain(example_facts()) and listing["kinds"]["equity"] == "money"
    assert listing["help"]["firm"]["runway_months"] and listing["help"]["policy"]["ltv"]


def test_a_request_is_answered_as_the_interface_answers_it(address):
    status, answer = post(address + "/api", json.dumps({"firm": plain(example_facts())}).encode())
    assert status == 200 and answer["ok"] and answer["read_out"]["equity"] == 20e6
    status, answer = post(address + "/api", json.dumps({"firm": {}}).encode())
    assert status == 200 and answer["ok"] is False and any("'firm' needs" in line for line in answer["problems"])


def test_what_is_not_a_request_is_refused_with_a_reason(address):
    status, answer = post(address + "/api", b"not json")
    assert status == 400 and "could not be read" in answer["problems"][0]
    status, answer = post(address + "/elsewhere", b"{}")
    assert status == 404 and "nothing at /elsewhere" in answer["problems"][0]
    with pytest.raises(urllib.error.HTTPError) as error:
        get(address + "/nothing")
    assert error.value.code == 404


def test_every_element_the_page_s_script_looks_for_exists():
    page = PAGE.read_text()
    markup, script = page.split("<script>", 1)
    wanted = set(re.findall(r'\$\("#([\w-]+)"', script))
    made_by_script = set(re.findall(r'\{id: "([\w-]+)"\}', script))
    present = set(re.findall(r'id="([\w-]+)"', markup))
    assert wanted and wanted - made_by_script <= present, wanted - made_by_script - present


def test_the_page_draws_only_with_what_it_carries():
    """No outside scripts, styles or fonts: it works with no network, and
    nothing about the firm leaves the machine."""
    page = PAGE.read_text()
    assert "http://www.w3.org/2000/svg" in page                    # a name, not a download
    assert not re.search(r'(src|href)\s*=\s*"https?://', page) and "@import" not in page
    assert 'fetch("/api"' in page and 'fetch("/inputs")' in page
    assert "innerHTML" not in page                                 # text goes in as text


def test_the_page_saves_a_firm_to_a_file_and_loads_one_through_the_model():
    """A saved firm is a request; loading posts it to the model, which checks
    it and fills in what the file leaves out, before anything on the page changes."""
    page = PAGE.read_text()
    for needed in ('id="save"', 'id="load"', 'id="load-file"', 'id="firm-name"', "function exportData()", "async function importData(saved)"):
        assert needed in page, needed
    assert 'await post({...saved, want: ["read_out"]})' in page
    assert "localStorage" not in page and "sessionStorage" not in page      # nothing is kept in the browser
