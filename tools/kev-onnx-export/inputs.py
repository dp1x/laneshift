"""The two inputs the report measures: a tiny state and a realistic 12-vehicle LANESHIFT-style state."""
import json

VEHICLES = [
    {"id": "ego", "x": 12.4, "y": 3.2, "v": 14.2, "lane": 1},
    {"id": "veh_01", "x": 26.8, "y": 3.1, "v": 11.7, "lane": 1},
    {"id": "veh_02", "x": 41.2, "y": 3.5, "v": 13.9, "lane": 1},
    {"id": "veh_03", "x": 8.1, "y": 6.9, "v": 15.4, "lane": 2},
    {"id": "veh_04", "x": 22.6, "y": 6.7, "v": 12.2, "lane": 2},
    {"id": "veh_05", "x": 37.9, "y": 7.0, "v": 14.8, "lane": 2},
    {"id": "veh_06", "x": -4.3, "y": 10.4, "v": 16.1, "lane": 3},
    {"id": "veh_07", "x": 17.5, "y": 10.6, "v": 10.9, "lane": 3},
    {"id": "veh_08", "x": 33.1, "y": 10.2, "v": 13.3, "lane": 3},
    {"id": "veh_09", "x": 3.8, "y": 0.2, "v": 9.8, "lane": 0},
    {"id": "veh_10", "x": 29.7, "y": 0.4, "v": 12.8, "lane": 0},
    {"id": "veh_11", "x": 44.5, "y": 0.1, "v": 11.1, "lane": 0},
]

TINY = {"state": "ego: 14.2 m/s lane 1 lead 14.4 m ahead",
        "questions": [{"instr": "should the ego vehicle brake?", "options": ["yes", "no"], "label": 0}]}

BIG_QUESTION = {"instr": "should the ego vehicle change lane to the left?", "options": ["yes", "no"], "label": 0}


def _state(pad=0):
    d = {"scenario": "urban arterial, 4 lanes, 12 vehicles, 2.5 s horizon",
         "vehicles": VEHICLES,
         "signals": [{"id": "sig_1", "state": "green", "t": 8.4},
                     {"id": "sig_2", "state": "red", "t": 2.1}],
         "pad": " ".join(["0"] * pad)}   # always present: its JSON overhead is part of the length budget
    return json.dumps(d)


def build(model, tok, target_tokens, question=BIG_QUESTION):
    """Encoding whose row (state + one branch) is exactly `target_tokens` long. max_state is lifted past encode()'s
    384-token training default: this is a serving-shaped input."""
    pad = 0
    while True:
        rec = {"state": _state(pad), "questions": [question]}
        enc = model.encode(tok, rec, max_state=4096, max_branch=8192)
        if len(enc["ids"]) >= target_tokens or pad > 200:
            break
        pad += 1
    if len(enc["ids"]) != target_tokens:
        raise RuntimeError(f"cannot hit {target_tokens} tokens with single-token pads; got {len(enc['ids'])} at pad={pad}")
    return enc
