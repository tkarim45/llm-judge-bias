"""Offline tests — deterministic fake judges, no keys, no network.

They verify the pair set is well-formed and that the bias arithmetic recovers a KNOWN injected bias:
a judge that always picks slot A must show first_slot_rate 1.0 and a 100% flip rate; a judge that
always picks the longer answer must show a high verbosity pick_worse_rate.
"""
from __future__ import annotations

from judgebias.benchmark import BiasFakeJudge, parse_choice, run
from judgebias.data import ASSERTIVENESS, POSITION, VERBOSITY, all_pairs


def test_pairs_are_wellformed():
    pairs = all_pairs()
    ids = [p.id for p in pairs]
    assert len(ids) == len(set(ids))                      # unique ids
    for p in pairs:
        assert p.bias in {"position", "verbosity", "assertiveness"}
        assert p.better and p.worse and p.better != p.worse
    # verbosity 'worse' really is the longer text
    for p in VERBOSITY:
        assert len(p.worse) > len(p.better)


def test_parse_choice():
    assert parse_choice("A") == "A"
    assert parse_choice(" B ") == "B"
    assert parse_choice("The better answer is A.") == "A"
    assert parse_choice("neither really") is None


def test_always_first_judge_shows_pure_position_bias():
    r = run(BiasFakeJudge(mode="always_first"), all_pairs())
    for bias in r["by_bias"].values():
        assert bias["first_slot_rate"] == 1.0            # always picks slot A
        assert bias["flip_rate"] == 1.0                  # verdict flips every time order swaps


def test_prefer_longer_judge_shows_verbosity_bias():
    r = run(BiasFakeJudge(mode="prefer_longer"), list(VERBOSITY))
    v = r["by_bias"]["verbosity"]
    # the longer answer is always the 'worse' (treatment) label -> pick_worse_rate ~ 1.0
    assert v["pick_worse_rate"] >= 0.9
    # picking purely by length is order-invariant -> no flips
    assert v["flip_rate"] == 0.0


def test_run_shapes():
    r = run(BiasFakeJudge(), all_pairs())
    assert set(r["by_bias"]) == {"position", "verbosity", "assertiveness"}
    for s in r["by_bias"].values():
        for k in ("pick_worse_rate", "first_slot_rate", "flip_rate"):
            assert 0.0 <= s[k] <= 1.0
    assert len(r["records"]) == len(all_pairs())
