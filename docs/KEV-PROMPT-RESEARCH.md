# Kev prompt research brief and GPT prompt

## What the model card actually says

Kev-0.8B is a **calibrated decision model**, not a text generator or a real-time
controller. It reads a state document and one or more typed questions, then
scores the supplied options:

- `choice`: 1–255 named options, each with an optional description
- `noul`: yes/no with optional descriptions
- `score`: ordered levels

It is a LoRA adapter plus a pointer head on Qwen3.5-0.8B-Base. The official
format is:

```text
<state> ...state...
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
```

The special tokens are the Qwen FIM/box tokens used by the Kev repo:
`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`,
`<|fim_suffix|>`. The current ONNX adapter uses the same tokens, so the
delimiter encoding is probably not the bug.

The model card says Kev-0.8B is intended for decisions close to its training
families (document classification, routing, policy checks over stated rules)
and is **markedly less accurate out of domain**. It recommends fine-tuning on
your own labels. The repo has a chess demo where the board is the state, legal
moves are `choice` options, and Kev plays or evaluates the position. A game
interface is therefore plausible, but it must look like a typed decision, not
like a 10 Hz control loop.

Current LANESHIFT prompt:

- state: a compact numeric JSON projection
- instruction: "Choose the safest useful next action from the currently legal choices. Only select an option present in criteria."
- options: `LEFT: ...`, `STAY: ...`, `RIGHT: ...`, `BRAKE: ...` with terse criteria
- fixed option order
- 10 Hz controller, 2–4 s inference, stale state
- rule-based safety heartbeat drives; Kev chooses `STAY` at ~0.38–0.41

## Working conclusion

The prompt is a major, testable cause. The model is being asked a policy/control
question in a format that under-specifies the safety rule, the lane geometry and
the option semantics. The current results do not distinguish:

1. a semantically bad prompt,
2. an out-of-domain model that needs fine-tuning,
3. a real-time latency/staleness failure.

The cheapest decisive experiment is an **offline prompt ablation** on logged
decision states, with the rule policy as a teacher and a small human-labelled
subset. Do that before more game runs or fine-tuning.

## Recommended next experiment

Use 100–300 logged decision states with a known safe action. Compare:

A. current JSON + terse options
B. natural-language scene + explicit safety rules + full option descriptions
C. chess-style state: lane occupancy plus legal actions, option descriptions
   state the immediate consequence
D. binary subquestions: "Is the left lane safe?", "Is the right lane safe?",
   "Must the ego brake?", then combine deterministically
E. option-order permutations of B or C, averaging probabilities per action
F. forced choice without `STAY` on hazardous states, to test whether `STAY`
   is a default/option-order artifact

Metrics: action accuracy vs rule/human, macro-F1, Brier, ECE, `STAY` rate,
unsafe-action rate, and whether a safe lane/brake is selected when the current
lane is blocked.

Go/no-go:

- If B/C/E raises safe-action selection and calibration, use that interface.
- If all variants remain near-uniform and `STAY`-biased, the bottleneck is the
  model/domain and fine-tuning is required.
- Do not evaluate with the safety heartbeat enabled. It hides the policy.

## The 4-hour plan (6–10 pm)

| Time | Work | Output |
| --- | --- | --- |
| 6:00–6:30 | Build natural-language prompt renderer from logged states; add lane gaps/TTC/rule sentences | prompt variants B/C |
| 6:30–7:15 | Label 100 held-out states (human or rule); run A/B/C/E; compute metrics | prompt comparison |
| 7:15–7:45 | Test binary subquestions or forced non-STAY; option-order permutation | best interface |
| 7:45–8:15 | Integrate best prompt into a decision-point mode; safety off | playable eval |
| 8:15–9:00 | Collect/approve human decisions; export Kev `choice` JSONL | fine-tune data |
| 9:00–9:45 | Fine-tune Kev-0.8B LoRA on Modal/GPU; held-out eval | checkpoint + metrics |
| 9:45–10:00 | Write result, limits, next step | report |

The fine-tune needs a GPU. The model card's recipe: start from
`jaredpalmer/kev-0.8b` with `--init_from`, lr `2e-5`, 1–2 epochs, batch 1 ×
accum 8 on a 4 GB GPU. The 0.8B base recipe is about 20 minutes on an H100;
small fine-tunes cost roughly $1 on Modal. Without a GPU, the 4-hour goal
should be the prompt ablation + data export, not a claimed fine-tune.

## Paste-ready GPT research prompt

```text
You are a senior ML research engineer. Research and design the fastest credible
way to make Kev-0.8B drive a simple 3-lane game, then fine-tune and compare it.
Think hard, cite the official sources, and end with a concrete 4-hour plan.

CONTEXT
- Project: LANESHIFT, a Godot 3-lane lane-runner. A Python WebSocket controller
  sends states; the game applies discrete LEFT/STAY/RIGHT/BRAKE actions.
- Goal: collect decisions, fine-tune Kev-0.8B, then compare base Kev,
  fine-tuned Kev, a rule policy and a small MLP on held-out seeds, with
  calibration and driving outcomes. The game is the testbed; humans, bots and
  scripted policies can all produce decisions.
- Current zero-shot result: Kev returns STAY 0.38-0.41, LEFT 0.17-0.24,
  RIGHT 0.19-0.22, BRAKE 0.16-0.24. STAY is the argmax every time. The rule
  safety fallback is doing the lane changes. Kev is not fine-tuned.
- Current prompt: a compact numeric JSON state; instruction "Choose the safest
  useful next action from the currently legal choices."; four options
  LEFT/STAY/RIGHT/BRAKE with terse criteria; fixed order; 10 Hz controller.
- Environment: Windows ARM64 CPU; ONNX Runtime; 2-4 s per decision; a local
  package. A GPU/Modal may be available for fine-tuning. Time budget: 4 hours,
  with a hard deadline. Human data collection is possible but limited.

OFFICIAL FACTS TO VERIFY AND USE
- Kev-0.8B is a calibrated decision model: a state plus typed questions
  (choice/noul/score), options with descriptions, a pointer head, no generated
  text. It is a LoRA adapter on Qwen3.5-0.8B-Base.
- Official format: <state> ... <q> instructions <opt> option </opt> ... <decide>.
  Special tokens are the Qwen FIM/box tokens.
- Model card: intended for document classification, routing, policy checks over
  stated rules; out-of-domain is markedly less accurate; fine-tuning on your
  own labels is recommended. The repo has a chess demo: board state, legal
  moves as Choice options, Kev plays or evaluates.
- Calibration temperature 2.35; option order can change answers; the serving
  API has a permutation endpoint.
- Fine-tune recipe: `--init_from jaredpalmer/kev-0.8b`, LoRA, lr 2e-5, 1-2
  epochs, batch 1 x accum 8 fits a 4 GB GPU; ~20 min for the 0.8B base recipe
  on an H100; small runs ~$1 on Modal.

QUESTIONS
1. Is the current prompt semantically wrong for Kev? Rank the likely causes:
   state representation, missing safety rule, terse option descriptions,
   fixed option order, four-way choice vs binary subquestions, real-time
   staleness, out-of-domain model, or missing fine-tune.
2. How should a driving decision be posed to Kev so it matches its training:
   exact state text, instruction, option names and descriptions? Provide 5
   prompt variants, including a natural-language scene and a chess-style
   legal-action prompt.
3. Should the action be a 4-way `choice`, a set of `noul` subquestions
   ("left safe?", "right safe?", "brake?"), or a `score`? Give the exact JSON
   request for each and the combination logic.
4. How should option order be handled? Is permutation averaging worth the
   latency? How many permutations?
5. Design an offline evaluation on 100-300 logged states with rule/human
   labels: metrics, split rules, go/no-go criteria, and how to avoid leakage.
6. Design the fine-tune dataset schema and the smallest credible fine-tune
   (records, labels, epochs, GPU, held-out split, calibration). What can be
   done without human labels, and what must wait?
7. Give a 4-hour execution schedule with exact commands/pseudocode and a
   fallback if no GPU is available. State what claim is allowed at the end.
8. Challenge the premise: is a game a good testbed for Kev? What would you do
   instead if the goal is a credible research result for an MBZUAI application?

OUTPUT
- A ranked diagnosis with evidence.
- Exact prompt templates and request JSON.
- A 4-hour plan with decision points.
- A fine-tune + evaluation plan with commands.
- Clear limits and a go/no-go recommendation.
Cite the official Kev model card/repo and any other sources you use.
```
