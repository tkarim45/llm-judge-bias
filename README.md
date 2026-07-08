# llm-judge-bias

**LLM-as-judge is the backbone of modern eval — and it has biases that inflate scores for reasons
that have nothing to do with quality.** This measures three of them (position, verbosity,
assertiveness) on controlled answer pairs, by judging each pair in *both* orders so the bias is
isolated from the correct verdict.

```bash
pip install -e ".[dev,real]"
llm-judge-bias                       # real Claude judge on Bedrock, every pair in both orders
llm-judge-bias --backend anthropic
llm-judge-bias --offline             # deterministic biased fake judge, no keys
pytest -q                            # offline tests — no keys, no network
```

## The setup

Each pair is engineered so a *fair* judge has an obvious, order-independent answer — any systematic
deviation is the bias:

- **position** — one answer is genuinely correct, the other genuinely wrong. A fair judge always picks
  the correct one, in either slot. Preferring whichever answer is shown *first* is position bias.
- **verbosity** — both answers are equally correct; one is concise, the other padded with true-but-
  filler detail. A fair judge is indifferent. Preferring the longer one is verbosity bias.
- **assertiveness** — both state the same correct facts; one is hedged ("I think it might be…"), the
  other confident. A fair judge scores them equal. Preferring the confident phrasing is style bias.

Every pair is judged twice (each answer in each slot), so `first_slot_rate` measures position
preference (0.50 = none) independently of `pick_worse_rate` (how often the judge chose the treatment:
the wrong / longer / confident answer).

## Results (real run — Claude Haiku 4.5 as judge on Bedrock)

| bias | pairs | pick_worse | first_slot | flip_rate | fair value |
|---|---:|---:|---:|---:|---|
| position | 12 | **0.000** | 0.500 | 0.000 | pick_worse ≈ 0.00 |
| verbosity | 10 | **0.750** | 0.550 | 0.100 | pick_worse ≈ 0.50 |
| assertiveness | 10 | **1.000** | 0.500 | 0.000 | pick_worse ≈ 0.50 |

`pick_worse` = how often the judge chose, for **position** the WRONG answer, for **verbosity** the
LONGER answer, for **assertiveness** the CONFIDENT phrasing. `first_slot` = how often it chose
whichever answer was shown first (0.50 = no position preference).

## Findings

- **No position bias — the "which is correct" judgment is rock-solid.** On pairs with a clear correct
  answer, the judge picked the wrong one **0%** of the time, chose the first slot exactly 50% of the
  time, and *never* flipped its verdict when the order was swapped. When one answer is genuinely
  better, Claude-as-judge finds it regardless of position. The bias is not in *correctness*.
- **The bias is in *style*. Verbosity bias: 75% preference for the longer answer.** Given two
  equally-correct answers, the judge preferred the one padded with true-but-redundant detail **75% of
  the time** (fair is 50%). Length reads as quality even when it adds no information.
- **Assertiveness bias is total: 100% preference for confident phrasing.** When the same correct facts
  were stated hedged ("I think it might be Canberra…") vs confidently ("It is definitely Canberra"),
  the judge picked the confident version in **every single pair**, in both orders. The judge equates
  confidence with correctness — even when the confident and hedged answers say exactly the same thing.
- **The practical consequence: LLM-as-judge scores are gameable.** A model being evaluated can lift
  its judge score by being longer and more assertive, with zero improvement in substance. Any
  eval, leaderboard, or RLAIF signal built on a naked LLM judge is measuring style as well as quality —
  which is why a serious judge needs order-swapping, length controls, and
  [calibration against human labels](https://github.com/tkarim45/llm-as-judge-system) before you trust
  its numbers.

> ⚠️ Scope: small controlled sets (12 position / 10 verbosity / 10 assertiveness pairs), one judge
> model, hand-authored pairs each designed to isolate one bias. The effects are large and clean, but
> the exact rates are directional — a different model, prompt, or scoring format (pointwise 1–10 vs
> pairwise A/B) will shift the magnitudes. The *shape* — robust on correctness, biased on style — is
> the reproducible result and matches the published LLM-judge-bias literature.

## How it works

```
src/judgebias/
  data.py        controlled pairs for each bias (correct-vs-wrong / concise-vs-padded / hedged-vs-confident)
  llm.py         Claude client (Bedrock default / direct Anthropic) + a deterministic biased fake judge
  benchmark.py   judge every pair in both slot orders → pick_worse / first_slot / flip rates per bias
  cli.py         run real (Bedrock/Anthropic) or --offline, optional --json
```

The offline path uses a fake judge with a *known* injected bias (always-first, or always-longer), and
the tests assert the metrics recover it — so CI verifies the bias arithmetic with no keys and no
network.

## License

MIT
