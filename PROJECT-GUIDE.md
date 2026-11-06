# LANESHIFT by Dhanesh and Rudraksh

**A research project on fine-tuning a small pretrained decision model to drive a 2D game.**

> **Central research question:** Can fine-tuning improve a pretrained decision model's performance in unseen scenarios?

**Short answer.** Yes for one narrow offline measure: the fine-tuned model predicted the teacher's labels better on unseen seeds. No for driving: it never learned to change lanes, never braked in the gameplay test, and both the original and fine-tuned models crashed on all 12 matched seeds. The project is a well-instrumented negative result about data coverage, rare manoeuvres, and the gap between offline accuracy and closed-loop control.

This guide is written for a school research project. It assumes no machine-learning background, but it does not hide the technical details. Every stage is explained at three levels.

| Level | Who it is for | What it contains |
|---|---|---|
| **SIMPLE** | Anyone | Plain English, analogies, the main idea |
| **TECHNICAL** | A technically curious student | What we built, how the parts fit together |
| **DEEP TECH** | An ML engineer | Terminology, architecture, equations, exact statistics |

The three levels describe the same work with increasing precision. If you want the story first, read the **SIMPLE** sections and the final note.

---

## Contents

1. [The Research Question](#1-the-research-question)
2. [The Game and Environment](#2-the-game-and-environment)
3. [The Original Kev Model](#3-the-original-kev-model)
4. [Human and Synthetic Data](#4-human-and-synthetic-data)
5. [Dataset Preparation](#5-dataset-preparation)
6. [LoRA Fine-Tuning](#6-lora-fine-tuning)
7. [ONNX Export](#7-onnx-export)
8. [Offline Evaluation](#8-offline-evaluation)
9. [Gameplay Comparison](#9-gameplay-comparison)
10. [Findings and Limitations](#10-findings-and-limitations)

- [Key Statistics at a Glance](#key-statistics-at-a-glance)
- [Reproducibility](#reproducibility)
- [Sources](#sources)
- [Detailed Reports and Artifacts](#detailed-reports-and-artifacts)

---

## The project in one diagram

```text
┌─────────────────────┐
│  Godot LANESHIFT    │  3 lanes, seeded traffic, 4 actions
│  driving simulator  │  records state + action + outcome
└──────────┬──────────┘
           │ telemetry and recordings
           ▼
┌─────────────────────┐
│  Demonstration data │  25,368 safe-teacher rows
│  human + synthetic  │  + 59 human rows
└──────────┬──────────┘
           │ exact Kev prompt, L256 token gate, seed-disjoint splits
           ▼
┌─────────────────────┐
│  25,427-row dataset │  16,795 train / 4,315 validation / 4,317 test
│  full dataset       │  test split locked away
└──────────┬──────────┘
           │ 4,096-row teacher-only subset, 1 epoch
           ▼
┌─────────────────────┐
│  LoRA fine-tune     │  Kev-0.8B + new adapter and pointer head
│  on A100 40 GB      │  512 optimizer steps, 41.1 minutes
└──────────┬──────────┘
           │ export
           ▼
┌─────────────────────┐
│  L256 ONNX model    │  runs on CPU inside the game controller
│  fine-tuned graph   │  new 1.507 GB fp16 weight blob
└──────────┬──────────┘
           │
           ├──────────────────────────────┐
           ▼                              ▼
┌─────────────────────┐        ┌─────────────────────┐
│  Offline evaluation │        │  Matched gameplay   │
│  4,317 test rows    │        │  12 fresh seeds     │
│  accuracy, F1, ECE  │        │  crashes, survival  │
└─────────────────────┘        └─────────────────────┘
```

---

## At a glance

| Stage | What we did | Main result |
|---|---|---|
| 1. Research question | Asked whether fine-tuning transfers to unseen scenarios | Two-part answer: offline yes, driving no |
| 2. Game and environment | Built a deterministic 3-lane Godot driving testbed | Reproducible simulator with full telemetry |
| 3. Original Kev | Integrated the released Kev-0.8B decision model | Ran locally, but drove poorly |
| 4. Data collection | Recorded human play and safe-teacher rollouts | 25,368 teacher + 59 human rows |
| 5. Dataset preparation | Built and validated seed-disjoint train/validation/test sets | 25,427 clean rows; 4,317-row untouched test |
| 6. LoRA fine-tuning | Trained on a 4,096-row subset for one epoch | Real checkpoint; loss fell; turns collapsed |
| 7. ONNX export | Exported the fine adapter and head to L256 ONNX | Real fine weights in a deployable graph |
| 8. Offline evaluation | Scored both models on the untouched test split | Accuracy 0.7160 → 0.8663; zero turns |
| 9. Gameplay comparison | Ran both models on 12 identical fresh seeds | Both crashed 12/12; no driving gain |
| 10. Findings | Audited the result and stated the limits | Negative result, clear next experiments |

---

## 1. The Research Question

| | |
|---|---|
| **What we did** | Defined the question, the hypotheses, and what would count as success. |
| **Why** | A fine-tune can look better on a test set and still be a worse driver. We wanted a result that measured both. |
| **How** | We tested offline action prediction and closed-loop gameplay on held-out seeds, using the same data and conditions for both models. |
| **Result** | Offline prediction improved narrowly. Closed-loop driving did not improve. Generalisation to new scenario families was not tested. |

### Simple

Imagine a student who has practised lots of multiple-choice questions. Now give them a new set of questions from the same subject. Did the practice help?

That is the basic idea. Kev is a small AI that answers multiple-choice questions about a driving scene. We fine-tuned it on examples of a careful driver. Then we gave it new roads it had never seen and asked two things:

1. Does it choose the same actions as the careful driver more often?
2. Does it actually drive better in the game?

The first question is like marking its exam paper. The second is like putting it in a car.

### Technical

The project is an applied imitation-learning experiment. Imitation learning means learning by copying demonstrated actions. We treat driving as a supervised classification problem: supervised because every example has a label.

- **State:** a text description of the road, the player, and nearby traffic.
- **Action:** one of four legal choices: `LEFT`, `STAY`, `RIGHT`, `BRAKE`.
- **Label:** the action taken by a human or by a rule-based teacher.
- **Model:** Kev-0.8B, a pretrained decision model that scores answer options.

We fine-tuned Kev on labelled state-action pairs, then evaluated it in two ways:

- **Offline:** compare predicted actions with the held-out labels.
- **Closed-loop:** let the model drive the game and measure crashes, survival, distance, score, overtakes, lane changes, and braking.

The primary hypothesis was that fine-tuning would improve both. A secondary hypothesis was that the small human block would add value beyond the synthetic teacher.

### Deep tech

A policy is a rule that maps a state to an action. Formally, we approximate a policy

$$
\pi_\theta(a \mid s)
$$

where $s$ is the schema-2 scene observation and $a \in \{\text{LEFT}, \text{STAY}, \text{RIGHT}, \text{BRAKE}\}$. Logits are the raw scores before probabilities; softmax turns them into a distribution that sums to 1. Kev parameterises this as a softmax over option logits produced by a pointer head:

$$
\pi_\theta(a_i \mid s) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Fine-tuning minimises the cross-entropy loss on the demonstrated action. Cross-entropy rewards giving high probability to the correct answer:

$$
\mathcal{L}(\theta) = -\log \pi_\theta(a^{\ast} \mid s)
$$

The experiment separates two quantities that are often conflated:

- **Offline risk:** $ \mathbb{E}_{(s,a) \sim D_{\text{test}}}[-\log \pi_\theta(a \mid s)] $, measured on held-out state-action pairs.
- **Closed-loop return:** the outcome of running $\pi_\theta$ in the simulator, where the model's actions change the future state.

A policy can improve the first while failing the second. That is exactly what happened.

**What "unseen scenarios" means here.** The test seeds were never used in training. They are new traffic layouts from the same six-cell scenario grid, with the same teacher, prompt, and action space. The test mixture is also reweighted toward density 1.5. This is an **unseen-seed** test, not an **unseen-scenario-family** test. We return to this distinction in the findings.

### Result

| Question | Answer |
|---|---|
| Did fine-tuning improve teacher-label prediction on unseen seeds? | Yes, narrowly and mostly through `STAY` and a subset of `BRAKE`. |
| Did fine-tuning improve driving on unseen seeds? | No convincing evidence. Both models crashed on every matched seed. |
| Did it generalise to new scenario families? | Not tested. The test uses the same six-cell grid. |
| Did the human data help? | Not tested at this scale. The 59 human rows were excluded from the fine-tune subset. |

---

## 2. The Game and Environment

| | |
|---|---|
| **What we did** | Built LANESHIFT, a 3-lane 2D driving game in Godot, with deterministic traffic and full telemetry. |
| **Why** | We needed a controlled environment where the same scenario can be replayed exactly and every decision can be recorded. |
| **How** | A kinematic simulator at 60 Hz physics, seeded traffic, four discrete actions, schema-2 observations, JSONL recordings. |
| **Result** | A working testbed that produced the entire dataset and both evaluation sessions. |

### Simple

LANESHIFT is a small driving game. You control a motorcycle on a three-lane road. Cars and trucks come toward you at different speeds. You can move left, stay in your lane, move right, or brake. If you hit a vehicle, the run ends.

The game is deliberately simple. That simplicity lets us replay the same road exactly, change one thing at a time, and record everything the AI sees and does.

### Technical

The game is built in **Godot 4.7.2**. The simulation runs at 60 physics ticks per second. A seed is a number that fixes the random choices, so a given seed always produces the same initial vehicles and the same spawn pattern. The player starts in the centre lane at 20 m/s, with a maximum speed of 28 m/s. Traffic moves at constant speeds between 13 and 21 m/s and never changes lane.

The model makes a decision about once per simulated second. Each decision produces one of four actions:

| Action | Effect |
|---|---|
| `LEFT` | Move one lane left, if legal |
| `STAY` | Keep the current lane and stop braking |
| `RIGHT` | Move one lane right, if legal |
| `BRAKE` | Slow down for 0.35 s |

The game is kinematic: vehicles move by scripted speeds and positions rather than a force-based physics engine. An episode is one run from start to crash or timeout. The legal action set is the game's own list of actions allowed at that moment. A token is a small chunk of text; the tokenizer splits the prompt into tokens. The game records three kinds of data:

- **Telemetry:** periodic state snapshots at about 30 Hz of simulation time.
- **Events:** decisions, applied actions, rejections, collisions, episode start/end.
- **Episode summaries:** score, distance, survival time, collisions, overtakes, action counts.

Every snapshot contains the player state, every active vehicle, the legal action set, and the current score. The model never sees pixels. It sees structured state.

### Deep tech

The simulation is kinematic:

- Player acceleration is $+3.0\ \text{m/s}^2$ when not braking, capped at 28 m/s.
- Braking decelerates at $-14.0\ \text{m/s}^2$, floored at 9 m/s.
- A lane change moves the player 4 m laterally at 15 m/s, so it takes about 0.27 s.
- Traffic is spawned ahead of the player at 30 + 18i metres, with a random lane, a ±4 m z-jitter, and a constant speed in [13, 21] m/s.
- Cars are 4.5 × 1.8 m; trucks are 7.0 × 2.4 m.
- Collision requires both a lateral overlap and a longitudinal overlap:
  $$
  |x_p - x_v| < \frac{w_v}{2} + 0.45
  \quad\text{and}\quad
  |z_p - z_v| < \frac{l_v}{2} + 0.8
  $$

The score is

$$
\text{score} = 0.1 \times \text{distance} + 5 \times \text{survival time} + 25 \times \text{overtakes}
$$

Distance is in metres, survival time in seconds, and overtakes is a count. This formula matters for the gameplay analysis: distance is a deterministic function of survival time, and score is dominated by survival. Only overtakes are partly independent.

The observation is **schema 2**, a versioned structured snapshot. The prompt renderer keeps up to nine safety-relevant vehicles:

1. The nearest vehicle ahead and behind in every lane.
2. Then the nearest remaining vehicles by distance.
3. The final `X<n>` field counts relevant vehicles omitted from the text.

The prompt is capped at 256 total tokens. For a four-option row, the maximum state portion is 154 tokens and the question branch costs 102 tokens, so 154 + 102 = 256. Rows with two or three legal actions can carry up to 159 state tokens because their option branch is shorter. This is the serving graph length, not the model's full context window.

A real prompt looks like this:

```text
v2
P C C 20 3 0 0 100 0 0 S
V laneDZ close abs type; +ahead -behind +closing; c car t truck
ON
L+33 +7 13 t
R+47 +2 18 c
R+51 +6 14 t
R+69 +2 18 t
OFF
none
L1/0 C0/0 R3/0 X0
```

The first line is the schema version. The `P` line describes the player. The `V` line defines the vehicle columns. `ON` and `OFF` separate visible from off-screen traffic. The last line counts vehicles by lane.

### Result

The game worked as a research testbed. It produced:

- 192 safe-teacher episodes.
- 3 human episodes.
- 25,427 usable dataset rows.
- Two matched gameplay sessions of 12 episodes each.

The simulator is simple, but it is deterministic, instrumented, and honest about what it measures.

---

## 3. The Original Kev Model

| | |
|---|---|
| **What we did** | Integrated the released Kev-0.8B decision model as the starting point and the "before" baseline. |
| **Why** | It is a pretrained decision model small enough to run on a laptop CPU, with open weights and a fine-tuning path. |
| **How** | Qwen3.5-0.8B-Base backbone, LoRA adapter, pointer head, scene prompt, L256 ONNX inference. |
| **Result** | It ran in the game, but it drove poorly: it was biased toward `STAY`, had low confidence, and crashed in most pilot runs. |

### Simple

Kev is a small AI that answers multiple-choice questions. You give it a description of a situation and a list of options. It gives you a probability for each option. It does not write an essay. It picks from the list.

The released Kev had never been trained on LANESHIFT. It knew how to make decisions in general, but it did not know this game. We used it as the starting point because fine-tuning a model that already understands decisions is easier than training one from scratch.

### Technical

Kev-0.8B is a **decision model**. It reads a state document and a set of typed questions, then returns a probability distribution over the supplied options in a single forward pass. It is built from three parts:

1. **A frozen language-model backbone:** Qwen3.5-0.8B-Base.
2. **A LoRA adapter:** small trainable matrices that adapt the frozen backbone.
3. **A pointer head:** a small network that scores each answer option.

A checkpoint is a saved copy of the model weights. The released checkpoint has about 11.3 million trainable parameters in the adapter and head. The base model has 0.8 billion parameters and stays frozen. This is why the model can run on modest hardware.

For the game, the original model was served through **ONNX**, a portable model format that runs without PyTorch. The game's Python controller loads the ONNX graph and tokenizer, builds the scene prompt, runs inference on the CPU, and sends one action back to Godot.

The released model's confidence was low. In one 60-second run, its probabilities were roughly:

| Action | Probability range |
|---|---:|
| `STAY` | 0.38–0.41 |
| `LEFT` | 0.17–0.24 |
| `RIGHT` | 0.19–0.22 |
| `BRAKE` | 0.16–0.24 |

`STAY` was the argmax every time. Argmax is the option with the highest probability. The model was choosing the least-bad option from a nearly uniform distribution.

### Deep tech

The backbone is Qwen3.5-0.8B-Base at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. It has 24 layers: 18 Gated DeltaNet layers (a linear-attention variant) and 6 full-attention layers. The released Kev checkpoint is `jaredpalmer/kev-0.8b@v1.0`.

The adapter is LoRA with:

- rank $r = 16$
- scaling $\alpha = 32$
- dropout 0.05
- 12 target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`
- 372 tensors, 10,822,656 parameters

The pointer head has 4 tensors and 524,800 parameters:

- `q.weight` [256, 1024], `q.bias` [256]
- `k.weight` [256, 1024], `k.bias` [256]

It projects the question's final token to a query and each option's closing token to a key, then scores the dot product scaled by $1/\sqrt{256} = 1/16$. A softmax over the legal options gives the action probabilities. Temperature scales the logits before softmax; a higher temperature makes the distribution less confident, but it does not change the argmax. The model is evaluated at temperature $T = 1.0$ in our experiments, even though the released checkpoint stores a fitted temperature of 2.3511.

The original model was evaluated in a small pilot: 5 seeds, 20 seconds each, traffic density 0.7, simulation speed 1.0. It crashed in 3 of 5 episodes and chose `STAY` in all 30 decisions. The rule-based baseline crashed in 0 of 5. This was the motivation for fine-tuning.

### Result

The original Kev ran end to end in the game. It was not a competent driver. It was slow on CPU, over-predicted `LEFT` on the test set, never predicted `RIGHT` or `BRAKE`, and had low confidence. It gave us a real baseline and a real starting checkpoint.

---

## 4. Human and Synthetic Data

| | |
|---|---|
| **What we did** | Collected human demonstrations and generated synthetic demonstrations from a deterministic safety teacher. |
| **Why** | Imitation learning needs many labelled examples of "in this situation, this action was taken." |
| **How** | One human session plus 192 safe-teacher episodes across six scenario cells, recorded as state-action pairs. |
| **Result** | 25,368 teacher rows and 59 usable human rows. The data had scale, but very few lane changes. |

### Simple

We needed examples of good driving. We got them in two ways.

First, a human played the game. That gives real human choices, but it is slow and produces only a small amount of data.

Second, we wrote a careful rule-following bot called the **safe teacher**. It checks the traffic ahead and behind, changes lanes only when it is safe, and brakes when no lane change is safe. It can play hundreds of episodes quickly.

The human data is like a few handwritten examples. The teacher data is like a large workbook. Both are useful, but they are not the same.

### Technical

**Human data.** One recording session produced three episodes. The session spanned about 12.5 hours of wall time, but only 395.7 seconds of simulated play. Almost all of the rest was idle time. The human made 61 decisions; 60 were accepted. We included 59 rows after excluding one unsafe lane change. The included rows are 29 `LEFT` and 30 `RIGHT`, with zero `STAY` and zero `BRAKE`. That is 0.23% of the full dataset.

The human rows are train-only. Validation and test are teacher-only. This prevents human behaviour from leaking into the held-out evaluation.

**Synthetic data.** The `safe_teacher` policy is deterministic. For the current lane and each adjacent lane, it computes the nearest vehicle ahead and behind, a front time-to-collision, and a rear clearance. It stays if the current lane is safe, moves to the safest adjacent lane if one is safe, and brakes otherwise.

The teacher played 192 episodes over six scenario cells. The human cell is shown for completeness:

| Cell | Density | Duration | Train seeds | Validation seeds | Test seeds |
|---|---:|---:|---:|---:|---:|
| A | 0.7 | 60 s | 8 | 2 | 2 |
| B | 1.0 | 60 s | 20 | 4 | 4 |
| C | 1.5 | 60 s | 22 | 10 | 10 |
| D | 1.5 | 90 s | 10 | 4 | 4 |
| E | 2.5 | 60 s | 38 | 6 | 6 |
| F | 3.5 | 60 s | 30 | 6 | 6 |
| H | 0.7 | human | 3 | 0 | 0 |

The teacher ran at 2.0× simulation speed, and the builder kept every second accepted decision, about 0.466 seconds of simulated time. This reduced near-duplicate frames while preserving the episode timeline.

The teacher is conservative, which is good for safety but bad for class balance. Most decisions are `STAY`. Lane changes are rare. The teacher crashed in 7 of 192 episodes (3.65%), so 518 pre-crash rows still carry `label_quality: ok`. The teacher is not perfect.

### Deep tech

The teacher's thresholds are:

- Front safety: gap ≥ 20 m or time-to-collision ≥ 3 s, with a hard floor of 10 m.
- Rear safety: gap ≥ 20 m or closing time-to-collision ≥ 2 s, with a hard floor of 8 m.
- A lane is safe only if both front and rear are safe.
- Among safe adjacent lanes, it picks the highest score, then the larger front gap, then `LEFT`.
- If no lane is safe, it brakes.

The full dataset action counts are:

| Action | Full dataset | Train | Validation | Test |
|---|---:|---:|---:|---:|
| `STAY` | 21,179 | 13,921 | 3,625 | 3,633 |
| `BRAKE` | 3,484 | 2,350 | 572 | 562 |
| `LEFT` | 378 | 262 | 61 | 55 |
| `RIGHT` | 386 | 262 | 57 | 67 |
| **Total** | **25,427** | **16,795** | **4,315** | **4,317** |

The teacher contributes 25,368 rows; the human contributes 59. Teacher turns are only 705 rows (2.78%). The 4,096-row subset later retained just 105 of them, all as isolated frames. That imbalance is the seed of the later failure.

The human session's 44,668 seconds of idle time is not a data-quality problem, but it is a useful reminder: wall-clock time is not the same as useful experience. The actual play was 0.88% of the session.

### Result

We had a large, clean synthetic dataset and a tiny human dataset. The synthetic data covered all six scenario cells and all five densities. It did not cover lane changes well, because the teacher rarely needed them and the human data was too small to compensate.

---

## 5. Dataset Preparation

| | |
|---|---|
| **What we did** | Converted recordings into Kev-format rows, enforced the token budget, and built seed-disjoint train/validation/test splits with provenance. |
| **Why** | Training data must match inference format exactly, and the test set must be genuinely held out. |
| **How** | Re-rendered every scene with the live encoder, kept only legal actions, rejected over-budget rows, split by seed, and validated every file. |
| **Result** | 25,427 clean rows across 195 seeds and episodes. The 4,317-row test split was never trained on. |

### Simple

A recording is a long list of "what happened." A training example is a single flashcard: "here is the road, here are the four choices, and here is the choice the driver made."

We turned the recordings into flashcards. We made sure every flashcard fits the model's 256-token limit. We split the flashcards by road seed, so the same road never appears in both training and test. Then we locked the test set in a drawer and did not touch it until the final evaluation.

### Technical

The builder is `python/build_kev_v2_scene.py`. It reads the preserved recordings and the seed plan. For every accepted decision, it:

1. Reconstructs the schema-2 observation.
2. Renders the exact scene prompt using the same `OnnxKevPolicy._encode` path used at inference.
3. Restricts the option set to the game's `legal_actions`, in the recorded order.
4. Checks the total token count against the 256-token serving graph.
5. Writes the row, provenance, and split assignment.

The provenance record is important. Provenance means metadata that records where each row came from. Each row carries its source (`human` or `safe_teacher`), session, episode, scenario seed, split, policy id, prompt hash, state hash, token count, label quality, and inclusion status. This makes the dataset auditable.

The split rule is **seed-disjoint**: every row from a given seed goes to exactly one split. This prevents near-duplicate leakage from the same road appearing in both training and test.

The builder excluded:

- 20 teacher rows over the 256-token budget.
- 1 unsafe human lane change, where the target lane had a vehicle 2.87 m ahead.

One cross-split prompt hash remains: a train row and a test row render byte-identically, even though their raw states differ. Both are `STAY`, both are 181 tokens, and the train row is inside the 4,096-row subset. It is 1 of 4,317 test rows (0.023%), so it cannot explain the `STAY`-heavy output.

The validator is `python/validate_kev_v2_scene.py`. It re-reads the JSONL and provenance files, checks schema, prompt hashes, legal actions, token budget, seed isolation, duplicate ids, and source labels. It reported no problems, with one duplicate-prompt warning.

### Deep tech

A Kev training row is a System One `choice` record:

```json
{
  "state": "...schema-2 scene prompt...",
  "questions": {
    "next_action": {
      "kind": "choice",
      "instructions": "Choose the safest legal action...",
      "criteria": {
        "LEFT": "Move left only with safe front and rear clearance.",
        "STAY": "Stay only if the current lane is safe.",
        "RIGHT": "Move right only with safe front and rear clearance.",
        "BRAKE": "Brake when no lane change is safe."
      },
      "label": 1
    }
  }
}
```

The label is the index into the ordered option list. The builder preserves the recorded legal-action order, so the label always points to the action the driver actually took. The data audit verified that the label is in the legal set on all 25,427 rows.

The token budget is a hard constraint:

- Maximum total tokens: 256.
- Minimum total tokens: 165.
- Maximum state tokens: 159 (four-option rows cap at 154; 2–3 option rows can carry more).
- Question/option branch: 102 tokens for four options.
- Over-budget rows: 0.

`--max_state 384` is Kev's training state-token cap, not a 384-token model context. All rows also fit the 256-token serving graph. Both checkpoints were evaluated at L256, so the comparison is length-matched.

The test split is **unseen seeds**, not unseen scenario types:

- Test seeds: 920001–920032.
- Same six-cell grid, same teacher, same prompt, same action space.
- Test rows: 4,317.
- Test action counts: 3,633 `STAY`, 562 `BRAKE`, 55 `LEFT`, 67 `RIGHT`.
- Test turn rows: 122 (2.83%).
- Test density 1.5 share: 47.6%, versus 27.4% in train.
- Cell/density total-variation distance: 0.2023. Total variation distance measures how different two distributions are; 0 means identical, 1 means no overlap.

Per-cell action distributions are close (TV 0.007–0.050), so the conditional policy is familiar. The scenario mixture is not identical.

### Result

The full dataset is real and internally clean:

| Split | Rows | Seeds | Episodes | Sources |
|---|---:|---:|---:|---|
| Train | 16,795 | 131 | 131 | 16,736 teacher + 59 human |
| Validation | 4,315 | 32 | 32 | 4,315 teacher |
| Test | 4,317 | 32 | 32 | 4,317 teacher |
| **Total** | **25,427** | **195** | **195** | **25,368 teacher + 59 human** |

All hashes verify. The test file hash `d16f65ab…` is unchanged. The dataset is good enough to train on and good enough to audit. The limitation is coverage: too few lane changes, too few human examples, and a test set from the same scenario family.

---

## 6. LoRA Fine-Tuning

| | |
|---|---|
| **What we did** | Fine-tuned Kev-0.8B on a 4,096-row teacher-only subset for one epoch on a rented A100. |
| **Why** | Full fine-tuning of a 0.8B model is expensive; LoRA adapts a small number of parameters and fits a short rental. |
| **How** | Rank-16 LoRA, bf16 compute, fp32 weights, batch 1 with 8-step accumulation, 512 optimizer steps. |
| **Result** | A real checkpoint with lower training loss, but only 105 fragmented turn examples. It learned `STAY` and some `BRAKE`, and no lane changes. |

### Simple

Fine-tuning is like giving the model a short course. We did not rebuild it from scratch. We froze most of its weights and trained a small set of extra knobs called **LoRA**. That is much cheaper.

We wanted to train on all 16,795 training rows for three epochs. That would have taken roughly 8 hours on the A100, far beyond the rental budget. So we selected a 4,096-row subset and ran one epoch. The subset was chosen to cover many seeds, episodes, densities, and all four actions, while keeping the natural action mix.

The run took 41 minutes. The loss went down. But the subset contained only 105 lane-change examples, and they were isolated frames rather than sequences. The model never learned to turn.

### Technical

**Why LoRA.** A full fine-tune updates all 0.8B weights. LoRA freezes the base model and adds low-rank matrices to selected layers. The rank controls the size of those matrices; rank 16 is small. The number of trainable parameters drops from 800 million to about 11.3 million. That makes the run cheaper, faster, and easier to store.

An epoch is one pass over the training data. Batch 1 with 8-step accumulation means the optimizer updates after eight examples. Gradient clipping caps the size of the update. FP32 is 32-bit floating point; BF16 is a 16-bit format with a wide exponent range.

**The subset.** The 4,096 rows came from the training split only, 24.4% of the 16,795 train rows. Selection rules:

- Teacher rows only. All 59 human rows were excluded to avoid teaching the human's weaving pattern.
- Crash episodes excluded.
- Duplicate prompt hashes dropped (the train split contained none).
- Up to 32 rows per seed, sampled at even intervals across the episode timeline.
- Natural action distribution, with no rare-action oversampling.

The subset covers 124 seeds and all densities 0.7–3.5:

| Action | Rows | Share |
|---|---:|---:|
| `STAY` | 3,439 | 83.96% |
| `BRAKE` | 552 | 13.48% |
| `LEFT` | 55 | 1.34% |
| `RIGHT` | 50 | 1.22% |

The training recipe:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B-Base` |
| Starting checkpoint | `jaredpalmer/kev-0.8b@v1.0` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | all 12 attention, MLP, and DeltaNet projections |
| Head dimension | 256 |
| Learning rate | 2e-5 |
| Head learning rate | 0.0, which Kev aliases to 2e-5 |
| Optimizer | AdamW, weight decay 0.01 |
| Schedule | OneCycleLR |
| Batch / accumulation | 1 / 8 |
| Epochs | 1 |
| Optimizer steps | 512 |
| Compute dtype | bf16 |
| Weight dtype | fp32 |
| Gradient clip | global norm 1.0 |
| Max state tokens | 384 |
| Seed | 1 |

**Why bf16 and fp32.** BF16 is a 16-bit floating-point format with a wide exponent range. It halves memory traffic and speeds up matrix multiplies on an A100. The LoRA weights stay in fp32 so the fine-tuned checkpoint can be merged and compared with the released fp32 checkpoint without an asymmetry. Kev does not support QLoRA or 8-bit training; the training path is bf16 compute over fp32 weights.

### Deep tech

LoRA replaces a weight matrix $W$ with

$$
W' = W + \frac{\alpha}{r} B A
$$

where $A$ and $B$ are low-rank matrices, $r = 16$, and $\alpha = 32$. Only $A$ and $B$ are trained. The base weights $W$ stay frozen. The Frobenius norm is the overall magnitude of a matrix.

The adapter has 372 tensors and 10,822,656 parameters. The pointer head has 524,800 parameters. Total trainable parameters: 11,347,456.

The run:

| Metric | Value |
|---|---:|
| Records | 4,096 |
| Optimizer steps | 512 |
| Wall time | 2,463.34 s (41.1 min) |
| Mean step | 4.811 s |
| Mean per record | 0.603 s |
| Forward tokens | 920,216 |
| Peak GPU allocation | 3.36 GB |
| Peak RSS | 5.67 GB |
| Loss | 0.812 → 0.393 |
| Adapter SHA-256 | `142e198a…` |
| Head SHA-256 | `7ed5d338…` |

The optimization was short and heavily clipped:

- 506 of 512 steps hit the global-norm clip of 1.0.
- Mean pre-clip gradient norm: 9.10.
- Maximum pre-clip gradient norm: 34.22.
- Head moved 1.236% of its Frobenius norm.
- Adapter moved 2.414% of its Frobenius norm.

This is a small effective update. It is not a global training failure, because the same optimizer learned `STAY` and a subset of `BRAKE`. It is a plausible contributor to the failure to re-rank a 1.2% turn class against an 84% `STAY` prior.

The subset's turn coverage is the leading explanation:

- 105 turn rows.
- 73 seeds.
- 51 seeds with exactly one turn.
- Maximum 4 turns per seed.
- 0 of 105 turns have another turn within four decisions in the same seed.
- 52 of the 125 teacher turn-bearing seeds in the full train split lost every turn.

The model saw each turn context once, as an isolated frame. For a model that cannot identify the conditional rule, predicting the class prior is a low-loss solution under cross-entropy. The fine model's turn probabilities, 1.24% `LEFT` and 1.42% `RIGHT`, are almost exactly the subset priors, 1.34% and 1.22%.

There is also a subset-selection defect. The top-up fallback over-weighted a few seeds: 122 seeds have 32 rows, seed 900001 has 128, and seed 900002 has 64. One train↔test duplicate prompt is inside the subset. These do not change the main conclusion, but they are real limitations.

### Result

The fine-tune is real. The adapter and head hashes differ from the original, the loss fell from 0.812 to about 0.393, and the checkpoint loads correctly. It learned `STAY` strongly and a high-precision subset of `BRAKE`. It did not learn `LEFT` or `RIGHT`.

The experiment used 4,096 rows, one epoch, and one training seed. The planned 3-epoch run on the full 16,795-row train split was not executed.

---

## 7. ONNX Export

| | |
|---|---|
| **What we did** | Exported the fine-tuned adapter and pointer head to an L256 ONNX graph with a new fp16 weight blob. |
| **Why** | The game runs ONNX Runtime on the CPU. It does not load PyTorch training checkpoints. |
| **How** | Loaded the fine checkpoint, wrote a new graph and external-data blob, verified hashes, and compared logits with PyTorch. |
| **Result** | A real fine-tuned graph that differs from the original. It runs in the game, at 1.581 s per decision versus 1.170 s for the original. |

### Simple

Training produces a PyTorch checkpoint. The game does not use PyTorch. It uses **ONNX**, a standard file format for neural networks, and **ONNX Runtime**, a small engine that runs the model.

We exported the fine-tuned model to ONNX so the game could drive with it. We did not reuse the original model's weight file. We wrote a new one from the fine checkpoint.

### Technical

The export produces three files:

- `kev07b_fp16_L256.onnx`: the graph, about 7 MB.
- `kev07b_fp16.data`: the weights, 1.507 GB, stored as an external fp16 blob. FP16 is 16-bit floating point, and the external data file keeps the weights separate from the graph.
- `tokenizer.json`: the tokenizer.

The graph is fixed at 256 tokens but has a dynamic option count. It can score any number of options, which is useful because the legal action set can be smaller than four.

The fine export was verified in three ways:

1. The graph and blob hashes differ from the original.
2. The graph structure hash differs from the original.
3. The fine ONNX logits are close to the fine PyTorch logits.

### Deep tech

Both graphs are opset 17 / IR 8. Opset is the ONNX operator set version. Inputs:

- `input_ids` [1, 256]
- `position_ids` [1, 256]
- `attention_mask` [1, 256]
- `decide_idx` [1]
- `opt_idx` [K], dynamic

Output:

- `logits` [K], dynamic

Both graphs use 324 external initializers in `kev07b_fp16.data`. The first is `lm.embed_tokens.weight` at offset 0, length 508,559,360 bytes.

Hashes:

| Artifact | SHA-256 |
|---|---|
| Fine ONNX graph | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` |
| Fine data blob | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` |
| Original ONNX graph | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |
| Original data blob | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` |

On the same 256-token probe:

```text
original ONNX logits [0.264047, 0.213320]  probs [0.512679, 0.487321]
fine ONNX     logits [0.378247, 0.421626]  probs [0.489157, 0.510843]
fine torch    logits [0.405219, 0.474689]
```

- Maximum fine-ONNX vs original-ONNX logit difference: **0.2083**.
- Maximum fine-ONNX vs fine-torch logit difference: **0.0531**.
- The argmax flips from option 1 for the original to option 2 for the fine model.
- The fine ONNX and fine torch agree on the argmax.

The 0.0531 difference is consistent with fp16 rounding in the exported weights. It is small, but it is a real numerical difference. The offline evaluation used the PyTorch path, not the ONNX path, so the reported offline metrics are torch metrics. A full ONNX test-set evaluation was not run.

### Result

The game can drive with the fine-tuned model. The exported graph is a genuine fine-tuned artifact, not a relabelled original. Its CPU latency is 1.581 s per decision, compared with 1.170 s for the original. That difference becomes important in the gameplay comparison.

---

## 8. Offline Evaluation

| | |
|---|---|
| **What we did** | Scored the original and fine-tuned checkpoints on the untouched 4,317-row test split. |
| **Why** | We needed to know whether fine-tuning improved action prediction and calibration on unseen seeds. |
| **How** | Same rows, same option order, same temperature, same precision path; accuracy, macro F1, log loss, Brier, and ECE. |
| **Result** | Accuracy rose from 0.7160 to 0.8663 and calibration improved sharply, but the fine model predicted zero `LEFT` and zero `RIGHT`. |

### Simple

This was the final exam. We gave both models the same 4,317 multiple-choice questions from roads they had never seen. We marked their answers.

The fine-tuned model scored higher. It also became much better calibrated: calibration means that when the model says it is 80% sure, it is right about 80% of the time. But there was a catch. It answered `STAY` almost every time. It never chose `LEFT` or `RIGHT`. A model that always stays in its lane can score well on an exam where most correct answers are `STAY`, while still being a bad driver.

### Technical

The test split is 4,317 rows from 32 unseen seeds. Both models saw the same records, the same option sets, the same option order, and the same evaluation temperature of 1.0.

The metrics:

| Metric | What it means |
|---|---|
| Accuracy | Fraction of rows where the top action matches the label |
| Macro F1 | Average of per-class F1 scores, treating each action equally |
| Log loss | Penalty for assigning low probability to the correct action |
| Brier score | Mean squared error of the predicted probabilities |
| ECE | Expected calibration error: how far confidence is from accuracy |
| Precision | Of the rows predicted as an action, the fraction that were correct |
| Recall | Of the rows that truly are an action, the fraction that were found |

Results:

| Metric | Original Kev | Fine-tuned Kev | Change |
|---|---:|---:|---:|
| Accuracy | 0.7160 | **0.8663** | +0.1503 |
| Macro F1 | 0.2143 | **0.3145** | +0.1002 |
| Weighted F1 | 0.7093 | **0.8228** | +0.1135 |
| Log loss | 0.9268 | **0.4093** | −0.5175 |
| Brier | 0.5417 | **0.2090** | −0.3327 |
| ECE | 0.2876 | **0.0395** | −0.2481 |
| Top-2 accuracy | 0.9166 | **0.9717** | +0.0551 |

The fine model wins on every headline metric. It also wins on accuracy and macro F1 in all 32 test seeds. The seed-clustered bootstrap gives:

- Accuracy delta: **+0.1503**, 95% CI **[+0.1080, +0.1962]**.
- Macro F1 delta: **+0.1002**, 95% CI **[+0.0789, +0.1232]**.

That looks strong until you compare it with a trivial baseline. The test set is 84.16% `STAY`. A model that always predicts `STAY` scores **0.8416** accuracy. The fine model scores 0.8663, only **+2.48 percentage points** above always-`STAY` (95% CI +1.98 to +3.13). Its macro F1 is 0.3145, versus 0.2285 for always-`STAY`.

Per-class results:

| Action | Support | Original P / R / F1 | Fine P / R / F1 |
|---|---:|---|---|
| `STAY` | 3,633 | 0.836 / 0.849 / 0.843 | 0.864 / 0.998 / 0.926 |
| `BRAKE` | 562 | 0.000 / 0.000 / 0.000 | 0.950 / 0.201 / 0.332 |
| `LEFT` | 55 | 0.008 / 0.091 / 0.015 | 0.000 / 0.000 / 0.000 |
| `RIGHT` | 67 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

The fine model predicted 4,198 `STAY`, 119 `BRAKE`, 0 `LEFT`, and 0 `RIGHT`. The original predicted 3,692 `STAY`, 625 `LEFT`, 0 `RIGHT`, and 0 `BRAKE`. 744 of 4,317 decisions changed.

### Deep tech

Accuracy is

$$
\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat{a}_i = a_i]
$$

Macro F1 is the unweighted mean of per-class F1:

$$
F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c},
\qquad
\text{macro F1} = \frac{1}{4}\sum_{c=1}^{4} F1_c
$$

ECE bins predictions by confidence and averages the gap between confidence and accuracy:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{|S_b|}{N} \left| \text{acc}(S_b) - \text{conf}(S_b) \right|
$$

The confidence intervals use a **seed-clustered bootstrap**: resample the 32 test seeds with replacement, not the 4,317 rows. Rows within a seed are correlated, so row-level standard errors understate uncertainty. ICC (intraclass correlation) measures how similar rows from the same seed are. The measured ICC is 0.055 for accuracy and 0.022 for macro F1; the design effect is how much that clustering inflates the variance, here 8.4 and 3.9. Row-level standard errors understate precision by about 2.6×.

The turn failure is precise:

- All 122 turn rows have the correct turn in the offered option set.
- The fine model assigns a nonzero probability to the correct turn on 122/122 rows.
- The correct turn is rank 3 or 4 on every turn row.
- Mean P(`LEFT`) on `LEFT` rows: 1.24%, range 0.63–2.60%.
- Mean P(`RIGHT`) on `RIGHT` rows: 1.42%, range 0.68–2.52%.
- Mean log-odds turn-vs-`STAY`: −4.34 for `LEFT`, −4.18 for `RIGHT`.
- These probabilities are essentially the subset priors: 1.34% `LEFT`, 1.22% `RIGHT`.

This is the signature of prior-fitting. The model learned the base rate of turns, not the conditions under which turns are safe.

Temperature is not the cause. The original stores T = 2.3511; the fine checkpoint stores T = 1.0. Both were evaluated at T = 1.0. Applying 2.3511 to either model changes zero argmax decisions, because temperature is monotonic and cannot reorder options.

### Result

The offline improvement is real and statistically robust. It is also narrow:

- `STAY` recall is 99.83%.
- `BRAKE` precision is 95.0%, but recall is only 20.1%.
- `LEFT` and `RIGHT` are never predicted.
- The gain over always-`STAY` is +2.48 percentage points.

The model became a better teacher-label predictor. It did not become a lane-changing driver.

---

## 9. Gameplay Comparison

| | |
|---|---|
| **What we did** | Ran the original and fine-tuned models on 12 identical fresh seeds in the real game. |
| **Why** | Offline accuracy does not tell us whether the model drives better. |
| **How** | Same seeds, duration, density, simulation speed, prompt, and L256 ONNX path; recorded crashes, survival, distance, score, overtakes, actions, and latency. |
| **Result** | Both models crashed on all 12 seeds. The fine model never changed lane and never braked. The small mean survival difference was not statistically convincing. |

### Simple

We put both models in the game on the same 12 new roads. Each road was played twice: once by the original Kev, once by the fine-tuned Kev.

Both models crashed every time. The fine-tuned model stayed in the centre lane for all 12 episodes and never braked. It survived about one second longer on average, but that average was carried by two unusual seeds. On a typical seed, it did worse.

### Technical

Matched conditions:

- Seeds: 930001–930012 (12 fresh seeds).
- Episode length: 60 s.
- Traffic density: 1.5.
- Simulation speed: 2.0×.
- Prompt: scene, schema 2.
- Model length: L256.
- Runtime: CPU ONNX, 8 threads.
- Same Godot build, project, and game mechanics.

Results:

| Metric | Original Kev | Fine-tuned Kev |
|---|---:|---:|
| Crashes | **12/12** | **12/12** |
| Completions | 0 | 0 |
| Mean survival | 6.886 s | 7.956 s |
| Median survival | **6.183 s** | 5.350 s |
| Mean distance | 182.3 m | 212.2 m |
| Median distance | **162.6 m** | 139.3 m |
| Mean score | 115.2 | 127.7 |
| Median score | **87.9** | 64.2 |
| Mean overtakes | 2.50 | 2.67 |
| Applied decisions | 21 | 19 |
| `LEFT` / `RIGHT` / `STAY` / `BRAKE` | 9 / 0 / 12 / 0 | **0 / 0 / 19 / 0** |
| Lane changes | **9** | **0** |
| Brake actions | 0 | 0 |
| Mean decision latency | **1.170 s** | 1.581 s |

The fine model stayed in lane 1 for all 12 episodes. The original attempted 10 left lane changes; 9 were applied and 1 was rejected. The raw rejection count is 3, not 5; an earlier summary CSV double-counted rejections. The corrected count is used here.

Paired differences, fine minus original:

| Metric | Mean difference | 95% CI | Median | Fine / Original / Tie | Sign p | Wilcoxon | Sign-flip |
|---|---:|---|---:|---|---:|---:|---:|
| Survival | +1.069 s | [−1.375, +3.708] | 0 | 5 / 4 / 3 | 1.000 | 0.570 | 0.461 |
| Distance | +29.94 m | [−38.50, +103.84] | 0 | 5 / 4 / 3 | 1.000 | 0.570 | 0.461 |
| Score | +12.51 | [−48.41, +72.83] | 0 | 5 / 4 / 3 | 1.000 | 0.734 | 0.734 |
| Overtakes | +0.167 | [−1.50, +1.75] | 0 | 3 / 2 / 7 | 1.000 | 1.000 | 0.938 |
| Lane changes | −0.750 | [−1.00, −0.50] | −1 | 0 / 9 / 3 | 0.0039 | 0.0034 | 0.0039 |
| Latency | +393.7 ms | [+240.2, +516.9] | +478 ms | 8 / 1 / 0 | 0.039 | 0.0078 | 0.0078 |

### Deep tech

The design is paired by seed. Each seed produces one original episode and one fine episode under the same initial conditions. The primary outcome is survival time. Secondary outcomes are distance, score, overtakes, lane changes, brake frequency, and latency.

Because both arms crash on every seed, the crash-rate comparison has zero variance. McNemar's test, which compares paired binary outcomes, is degenerate. The continuous outcomes are analysed with:

- A paired sign test.
- A Wilcoxon signed-rank test, which compares paired continuous outcomes.
- An exact sign-flip permutation test, which randomly flips the sign of each paired difference to build a null distribution.
- A paired bootstrap confidence interval.

The mean survival difference is +1.069 s, but the median is 0 and the win/loss/tie count is 5/4/3. The sign test p-value is 1.00. The sign-flip p-value is 0.46. The 95% CI includes zero.

The mean is not robust. Seeds 930005 and 930012 account for **137%** of the survival sum. Without those two seeds, the mean difference is **−0.48 s**. Without the top three, it is **−1.23 s**. A result carried by two outliers is not evidence of a general improvement.

The study is also underpowered. At n = 12, the minimum detectable effect (MDE) for survival is **3.82 s**. The observed +1.07 s is 28% of the MDE. About **154 seeds** would be needed for 80% power at the observed effect. The 12-seed result cannot distinguish a real small improvement from noise.

Distance and score are not independent evidence. Distance is a deterministic function of survival time in this simulator (r = 1.0000), and score is dominated by survival (r = 0.959). Only overtakes are partly independent, and their difference is not significant.

The latency confound is material. The original model averaged 1.170 s per decision; the fine model averaged 1.581 s. The game applies an action only if it is fresh enough. The original received 24 decision requests and applied 21; the fine received 19 and applied 19. The original's lane changes were executed on observations about 2.4 s stale. The fine model had fewer control opportunities. This is a comparison of the deployed CPU systems, not an isolated comparison of the two policies.

Three fine episodes received no applied control at all. `STAY` is a physical no-op, so those episodes were effectively uncontrolled. On the three seeds where the original also issued only `STAY`, the outcomes are bit-identical despite different decision counts. That is consistent with the game being deterministic and `STAY` changing nothing.

### Result

Neither model is a competent driver. Both crash on every matched seed. The fine model has a slightly better mean survival, distance, and score, but a worse median survival and median score. It never changes lane and never brakes. The original is more active and more error-prone. The difference is not statistically convincing, and it is confounded by latency and control opportunity.

This is not evidence that the fine-tune drives better.

---

## 10. Findings and Limitations

| | |
|---|---|
| **What we did** | Consolidated the experiment, audited the result, and separated verified facts from interpretations and open questions. |
| **Why** | A negative result is only useful if it is honest about what was measured and what was not. |
| **How** | Three independent read-only audits plus an independent recalculation of the statistics. |
| **Result** | Fine-tuning improved offline teacher-label prediction and calibration, but did not teach lane changes and did not improve driving. |

### Simple

The model got better at the exam and no better at driving.

That sounds strange, but it makes sense. Most of the exam answers were `STAY`. The fine-tuned model learned to say `STAY` almost every time, and it learned to spot some braking situations. That raised its score. But it never learned when to change lanes, because the training data contained very few lane changes, and they were scattered as isolated frames.

In the game, a driver that never changes lane and never brakes will eventually hit traffic. Both models did, on every seed.

### Technical

The research question has a two-part answer.

| Question | Answer | Evidence |
|---|---|---|
| Does fine-tuning improve action prediction on unseen seeds? | Yes, narrowly | Accuracy 0.7160 → 0.8663; fine wins 32/32 seeds; cluster CI above zero |
| Does it improve driving on unseen seeds? | No convincing evidence | Both crash 12/12; survival +1.07 s, CI includes zero; fine never turns or brakes |
| Does it generalise to unseen scenario families? | Not tested | Test is unseen seeds on the same six-cell grid, reweighted toward density 1.5 |
| Does human data help? | Not tested at this scale | 59 human rows (0.23%), excluded from the fine-tune subset |

The offline gain is real but narrow. The fine model's accuracy is only +2.48 percentage points above a constant-`STAY` predictor. Macro F1 improves from 0.2143 to 0.3145, almost entirely through `BRAKE`. `LEFT` and `RIGHT` F1 are zero.

The driving result is a null. Both models crash on all 12 seeds. The mean survival difference is +1.069 s with a 95% CI of [−1.375, +3.708]. The median difference is 0. The sign test p-value is 1.00. Two seeds carry 137% of the mean; without them, the mean is negative.

### Deep tech

**Why offline accuracy improved.** The test set is 84.16% `STAY`. A constant-`STAY` predictor already scores 0.8416. The fine model scores 0.8663. The gain comes from three things:

1. Strong `STAY` recall: 0.9983.
2. A high-precision `BRAKE` detector: 0.9496 precision, 0.2011 recall.
3. Better calibration: ECE 0.2876 → 0.0395.

`BRAKE` has 552 rows in the subset, 5.26× the 105 turn rows, and 116 seeds versus 73. The expected number of optimizer steps containing at least one `BRAKE` example is 351, versus 96 for turns. That is enough signal to learn a partial `BRAKE` policy. It is not enough to learn a rare, state-conditioned lane change.

This is a textbook behaviour-cloning failure mode: the policy learns the marginal action distribution, `STAY`, but not the rare state-conditioned manoeuvre.

**Why lane changes collapsed.** The evidence supports three contributing factors, with one leading explanation:

1. **Rare, fragmented coverage (leading).** 105 turn rows across 73 seeds; 51 singletons; max 4 per seed; 0/105 with a neighbouring turn within four decisions; 52/125 turn-bearing seeds dropped. The model never saw a manoeuvre as a sequence.
2. **Short, clipped update (contributing).** One epoch, 512 steps, 506/512 clipped, mean pre-clip gradient norm 9.10. The head moved 1.24% and the adapter 2.41% of their Frobenius norms. This is a small effective update.
3. **Original head prior (context).** The released head already over-predicted `LEFT` and never predicted `RIGHT` or `BRAKE`. Fine-tuning suppressed `LEFT` into `STAY` but did not re-rank turns above `STAY`.

Labels, legal masks, decoding, loss, checkpoint loading, and ONNX export were all checked and found correct. The zero-turn result is behavioural, not an implementation bug. Temperature is not the cause: it cannot reorder options.

**Why driving did not improve.** The fine model never turns and never brakes. `STAY` is a physical no-op. In a deterministic simulator with traffic ahead, a no-op policy eventually crashes. The gameplay mean is not robust, the sample is underpowered, and the latency difference gives the two models different control opportunities.

**What is defensible.**

- The dataset counts, hashes, labels, splits, and provenance.
- The zero-turn behaviour on the 4,317-row test set.
- The offline metrics and the 32/32 seed improvement.
- The fine model beats the original offline.
- The fine model beats constant-`STAY` by a small margin.
- The fine ONNX graph and blob differ from the original.
- Both gameplay arms crash 12/12.
- The latency difference is real.
- The test is unseen seeds, not unseen scenarios.

**What is not defensible.**

- That fine-tuning improves driving.
- That it generalises to novel scenario families.
- That the cause of the zero-turn result is proven.
- That the human data helped.
- That the model is safe or deployable.
- That the marginal gameplay mean is a real improvement.

**Limitations.**

| Limitation | Why it matters |
|---|---|
| One training run, one seed | No estimate of run-to-run variance |
| One epoch, 512 steps | One third of the planned schedule |
| 4,096-row subset | 24.4% of the 16,795-row train split; not the full dataset |
| Teacher-only subset | Human data's effect is untested |
| 105 fragmented turns | The model never sees a manoeuvre sequence |
| Subset seed imbalance | 900001 has 128 rows, 900002 has 64 |
| One train↔test duplicate prompt | Minor leakage; both rows are `STAY` |
| 7 teacher crashes in 192 episodes | 518 pre-crash rows ship as `ok` |
| Human data 0.23% | Too small to support a human-vs-synthetic claim |
| Test is same scenario family | No unseen scenario-family holdout |
| Test mixture reweighted | Density 1.5 is 47.6% of test vs 27.4% of train |
| 12 gameplay seeds, all crashes | Low power; MDE 3.82 s |
| Latency mismatch | 1.170 s vs 1.581 s; different control opportunities |
| Tokenizer files differ | Vocab 248,066 vs 248,077; state tokens matched 41/41 logged states |
| ONNX parity 0.053 logits | fp16 rounding; same argmax on the probe |
| No full ONNX test evaluation | Offline metrics are torch metrics |
| Temperature reset | Fine head stores T=1.0; original stores T=2.3511 |
| Simple 2D kinematic game | Not a realistic driving simulator |

**What we would do next.**

| Experiment | Why |
|---|---|
| Turn/brake-enriched, context-clustered dataset | Keep all 465 teacher turns and contiguous turn sequences; add 2–3 epochs; hold out a scenario family |
| DAgger / on-policy correction | DAgger (Dataset Aggregation) lets the model drive and has the teacher or human correct it around lane changes and braking |
| Latency-matched closed-loop benchmark | Fix the decision interval above both latencies; use ≥154 seeds; ablate head LR, class weighting, LoRA targets, and human-vs-synthetic mixture |

### Result

The experiment is a clean negative result. Fine-tuning improved teacher-label prediction and calibration on unseen seeds. It did not teach lane changes and did not improve driving. The most useful finding is the size of the gap between offline accuracy and closed-loop competence, and the evidence that rare, fragmented manoeuvre coverage plus a short clipped update can produce a policy that looks better on paper and behaves no better on the road.

Do not deploy this checkpoint.

---

## Key Statistics at a Glance

| Area | Number |
|---|---|
| Full dataset | 25,427 rows: 16,795 train / 4,315 validation / 4,317 test |
| Sources | 25,368 safe-teacher + 59 human |
| Seeds / episodes | 195 |
| Test seeds | 32 unseen seeds, 920001–920032 |
| Test scenario family | Same six-cell grid; density 1.5 is 47.6% of test vs 27.4% of train; TV 0.2023 |
| Actual fine-tune data | 4,096 rows, 124 seeds, teacher-only, 24.4% of the train split |
| Subset turns | 105 rows (55 `LEFT`, 50 `RIGHT`), 73 seeds, 51 singletons, max 4/seed |
| Training | 1 epoch, 512 steps, 41.1 min, A100 40 GB, about 4.6 AED |
| Training loss | 0.812 → 0.393 |
| Gradient clipping | 506/512 steps clipped; mean pre-clip norm 9.10 |
| Trainable parameters | 11,347,456 (10,822,656 adapter + 524,800 head) |
| Offline accuracy | 0.7160 → 0.8663 |
| Constant-`STAY` accuracy | 0.8416; fine is +2.48 pp above it |
| Macro F1 | 0.2143 → 0.3145 |
| ECE | 0.2876 → 0.0395 |
| Fine predictions | 4,198 `STAY`, 119 `BRAKE`, 0 `LEFT`, 0 `RIGHT` |
| Offline seed wins | 32/32 on accuracy and macro F1 |
| Gameplay | 12 matched seeds, both models crash 12/12 |
| Gameplay survival | +1.069 s, 95% CI [−1.375, +3.708], median 0 |
| Gameplay actions | Fine: 0 lane changes, 0 brakes; original: 9 lane changes |
| Gameplay latency | 1.170 s original vs 1.581 s fine |
| Power | MDE 3.82 s at n=12; about 154 seeds for 80% power |

---

## Reproducibility

The project is published on GitHub at [dp1x/laneshift](https://github.com/dp1x/laneshift), branch `feature/portable-windows-arm64`.

### 1. Get the code

```powershell
git clone https://github.com/dp1x/laneshift.git
cd laneshift
git checkout feature/portable-windows-arm64
```

### 2. Install the game and Python dependencies

- Godot 4.7.2 Stable.
- Python 3.10 or newer.

```powershell
python -m pip install -r python/requirements.txt
```

### 3. Run the tests

```powershell
run_tests.bat
```

Or run the Python and Godot checks separately:

```powershell
cd python
python -m unittest discover -s tests -v
```

### 4. Validate the dataset

```powershell
python python/validate_kev_v2_scene.py --root datasets/kev-v2-scene
```

This checks schema, prompt hashes, legal actions, token budget, seed isolation, duplicate ids, and source labels.

### 5. Reproduce the fine-tune

The actual run used the 4,096-row subset, one epoch. Copy `datasets/kev-subset-4096/subset_train.jsonl` to `/workspace/data/subset_train.jsonl` on the GPU host:

```bash
python -m kev.train \
  --data /workspace/data/subset_train.jsonl \
  --base Qwen/Qwen3.5-0.8B-Base \
  --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --init_from jaredpalmer/kev-0.8b@v1.0 \
  --lora 16 --head_dim 256 --lora_targets all \
  --lr 2e-5 --epochs 1 --batch 1 --accum 8 \
  --dtype bf16 --weights_dtype fp32 --checkpointing 1 \
  --max_state 384 \
  --p_none 0 --p_none_distract 0 --p_distract 0 --p_none_pair 0 \
  --device cuda --seed 1 \
  --out /workspace/runs/laneshift-kev-subset-e1
```

Expected time on an A100 40 GB: about 41 minutes for 4,096 rows, one epoch.

The handoff default is 3 epochs on the full 16,795-row train split. That run was not executed. If you reproduce it, keep `--weights_dtype fp32` and evaluate both checkpoints at the same L256 length.

### 6. Reproduce the offline evaluation

Use `notebooks/kev_finetune.ipynb`. The notebook:

- loads the original and fine checkpoints,
- scores both on the same 4,317 test rows,
- uses the same option order and temperature,
- writes `eval_metrics.json` and per-row predictions.

The detailed results are in [`research/audits/RESULTS.md`](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/RESULTS.md).

### 7. Reproduce the gameplay comparison

The ONNX weights are release assets, not ordinary Git files. Download and reassemble them first. Instructions are in [`research/model-artifacts/EXTERNAL-ARTIFACTS.md`](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/model-artifacts/EXTERNAL-ARTIFACTS.md).

Then run the fine model on the 12 matched seeds:

```powershell
python python/experiment_runner.py `
  --policies onnx_kev `
  --seeds 930001,930002,930003,930004,930005,930006,930007,930008,930009,930010,930011,930012 `
  --duration 60 --traffic-density 1.5 --sim-speed-scale 2.0 `
  --onnx-model models\kev-finetuned-subset-e1\kev07b_fp16_L256.onnx `
  --onnx-tokenizer models\kev-finetuned-subset-e1\tokenizer.json `
  --onnx-length 256 --onnx-threads 8 `
  --onnx-model-id "laneshift-kev-v2-subset-e1+onnx:L256" `
  --output fine-12.csv
```

Repeat with the original model files and a different output path. Compare paired seeds.

### 8. Check hashes

Every published artifact has a SHA-256 in the release notes, the dataset `SHA256SUMS`, or the final report. The key hashes:

| Artifact | SHA-256 |
|---|---|
| Fine adapter | `142e198a66fde6204d85bcd42be9f8170cbcf4741fcc5a739184b42004a90ba8` |
| Fine head | `7ed5d33839f8e1aedc3322a5c186fcd4c4d3dbba7ac30d992828d9e8cff6bc38` |
| Fine ONNX | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` |
| Fine ONNX data | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` |
| Test split | `d16f65ab244b587ad0b7a56e702eb426543dedd4f7c5e4548af9d7bf58471da4` |

---

## Sources

These are the models, tools, and resources the project actually used.

- **Kev-0.8B (original model)** — [Hugging Face](https://huggingface.co/jaredpalmer/kev-0.8b) · [GitHub](https://github.com/jaredpalmer/kev). The pretrained decision model we started from. It is a LoRA adapter plus pointer head on Qwen3.5-0.8B-Base. Its model card supplied the architecture and training details. Used as the starting checkpoint and the "before" baseline.
- **Qwen3.5-0.8B-Base** — [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base). The base model and tokenizer underneath Kev, loaded during fine-tuning. Its model card supplied the architecture details.
- **Kev codebase** — [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev). Training (`kev.train`), model code, and serving tools. Pinned to commit `2074cc44256200bb37540103fa4572901faa10f3`.
- **Godot 4.7.2** — [godotengine.org](https://godotengine.org/). Built the LANESHIFT driving simulator, recording, telemetry, and controller bridge.
- **Vast.ai** — [vast.ai](https://vast.ai/). Rented one A100 PCIe 40GB for the fine-tune. About 2 hours at $0.608/hr, roughly 4.6 AED.
- **Fine-tuned Kev-0.8B (ours)** — [GitHub release](https://github.com/dp1x/laneshift/releases/tag/kev-0.8b-laneshift-v2-subset-e1). The new adapter, pointer head, and L256 ONNX export from the 4,096-row, one-epoch run. Used in the offline and gameplay comparisons.
- **PyTorch 2.8.0 + CUDA 12.8** — [pytorch.org](https://pytorch.org/). Training, evaluation, and ONNX export.
- **Hugging Face Transformers** — [docs](https://huggingface.co/docs/transformers/index). Model loading and training loop.
- **Hugging Face PEFT** — [docs](https://huggingface.co/docs/peft/index). LoRA adapters.
- **Hugging Face Accelerate** — [docs](https://huggingface.co/docs/accelerate/index). Device and mixed-precision handling.
- **Hugging Face Datasets** — [docs](https://huggingface.co/docs/datasets/index). Data loading.
- **ONNX Runtime** — [onnxruntime.ai](https://onnxruntime.ai/). CPU inference for the exported L256 graph inside the game controller.
- **Python** — [python.org](https://www.python.org/). Dataset builders, experiment runner, and evaluation scripts.
- **NumPy** — [numpy.org](https://numpy.org/). Statistics and data handling.
- **LANESHIFT project** — [repository](https://github.com/dp1x/laneshift/tree/feature/portable-windows-arm64) · [dataset](https://github.com/dp1x/laneshift/tree/feature/portable-windows-arm64/datasets/kev-v2-scene) · [final report](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/FINAL-REPORT.md). Code, the 25,427-row dataset, the 4,096-row subset, and all results.

Kev and Qwen3.5 are published under Apache-2.0. The LANESHIFT repository does not yet declare a license.

---

## Detailed Reports and Artifacts

The main guide summarises the result. These reports contain the full evidence.

| Report | What it covers |
|---|---|
| [Final report](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/FINAL-REPORT.md) | Full synthesis, 14 questions, verified facts vs interpretations |
| [Data audit](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/DATA-AUDIT.md) | All 25,427 rows, hashes, provenance, splits, teacher outcomes |
| [Model audit](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/MODEL-AUDIT.md) | Checkpoint loading, LoRA/head tensors, loss, labels, ONNX parity |
| [Statistics audit](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/STATISTICS-AUDIT.md) | Offline metrics, seed-clustered bootstrap, gameplay tests |
| [Gameplay comparison](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/GAMEPLAY-COMPARISON.md) | Matched 12-seed original-vs-fine results |
| [Turn analysis](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/TURN-ANALYSIS.md) | Why lane changes failed |
| [Results](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/audits/RESULTS.md) | Training and offline metrics |
| [Dataset](https://github.com/dp1x/laneshift/tree/feature/portable-windows-arm64/datasets/kev-v2-scene) | Full 25,427-row dataset and cards |
| [Subset](https://github.com/dp1x/laneshift/tree/feature/portable-windows-arm64/datasets/kev-subset-4096) | Exact 4,096-row training subset |
| [External artifacts](https://github.com/dp1x/laneshift/blob/feature/portable-windows-arm64/research/model-artifacts/EXTERNAL-ARTIFACTS.md) | Model release assets and reassembly instructions |
| [Release](https://github.com/dp1x/laneshift/releases/tag/kev-0.8b-laneshift-v2-subset-e1) | Adapter, head, ONNX graph, and weight blobs |

---

## A Final Note

The most useful result in this project is the gap between the accuracy number and the driving.

A model can look better on a held-out test set and still be a worse driver. That happens when the test set is dominated by one easy action, when the rare action is barely represented, and when the training signal is too short to learn the rare action's conditions. LANESHIFT measured that gap directly.

The next experiment should target lane-change coverage and closed-loop evaluation. More accuracy on the same distribution will not answer the question.
