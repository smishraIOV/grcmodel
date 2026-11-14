"""What each input means, for a person filling in a form: every input has
an entry, and nothing in the guide is about an input that does not exist.
"""

import pytest

from quant.api import inputs
from quant.guide import GUIDE, guide, names, resolve
from scripts.dashboard import PAGE


def test_every_input_has_a_label_a_meaning_and_what_to_enter():
    for part in GUIDE:
        for name in names(part):
            label, meaning, enter = resolve(part, name)
            assert label and meaning and enter, (part, name)
            assert len(label) <= 60 and 20 <= len(meaning) <= 420 and len(enter) <= 260, (part, name, len(meaning), len(enter))
            assert meaning.endswith(".") and enter.endswith("."), (part, name)


def test_the_guide_is_about_inputs_that_exist():
    for part, entries in GUIDE.items():
        known = set(names(part))
        shared = {"*." + name.split(".", 1)[1] for name in known if "." in name}
        assert set(entries) <= known | shared, set(entries) - known - shared


def test_the_parts_of_a_statement_are_covered():
    assert "small_raise.amount" in names("firm") and "loan_demand.rate" in names("market")
    assert resolve("risk", "grc_on_incidents.reduction") == resolve("risk", "grc_on_losses.reduction")   # shared by the four
    assert resolve("firm", "small_raise.cost") != resolve("firm", "large_raise.cost")
    assert names("rule") == ["years", "equity_buffer", "largest_raise"]


def test_the_two_the_user_asked_about_say_how_to_read_the_number():
    crashes = resolve("market", "btc_crashes_per_year")
    assert "rate, not a percentage" in crashes[1] and "0.5 means one crash every two years" in crashes[2]
    view = resolve("market", "btc_view")
    assert "in the firm\'s opinion" in view[1] and "0% means no view" in view[2]


def test_the_interface_and_the_page_carry_the_guide():
    listing = inputs()
    assert listing["guide"] == guide() and set(listing["guide"]) == {"firm", "market", "risk", "policy", "rule"}
    for part in ("firm", "market", "risk", "policy"):
        assert set(listing[part]) <= set(listing["guide"][part]), part       # every input shown has an entry
    assert listing["units"]["btc_crashes_per_year"] == "/yr"
    page = PAGE.read_text()
    assert "META.guide[part]" in page and "function showHelp(anchor, entry)" in page and 'class: "info"' in page


@pytest.mark.parametrize("part", ["firm", "market", "risk", "policy"])
def test_labels_within_a_part_are_distinct_enough_to_tell_inputs_apart(part):
    labels = [resolve(part, name)[0] for name in names(part) if "." not in name]
    assert len(set(labels)) == len(labels)
