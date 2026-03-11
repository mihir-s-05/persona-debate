# debate-v-majority

Evaluation harness for comparing LLM inference strategies on reasoning benchmarks. Runs single-shot, majority-vote, and multi-agent persona-conditioned debate with a judge, producing structured JSONL output for reproducible analysis.

## Process overview

1. **Initial solve** -- each persona independently receives the question and solves it, unaware of the debate or voting.
2. **Majority vote** -- answers are parsed and a simple mode is computed. If no answer is repeated, the result is `None`.
3. **Debate rounds** -- each persona sees every other persona's visible output (thinking excluded) and produces an updated response. This repeats for `k` rounds.
4. **Final majority vote** -- another mode is computed on the final-round outputs.
5. **Judge evaluation** -- a judge persona receives the full debate transcript (all rounds, all agents, visible output + thought summaries) and selects the best answer based on argument quality. The judge does not re-solve the problem.

## Persona system

Personas are generated per-question along diversity axes to cover the solution space:

- **Fixed axes** (default 4) enforce problem-solving methodology diversity across all questions.
- **Task axes** (default 2) are generated with the specific problem's context.
- **5 personas** are sampled via maximin spacing across the resulting 6-dimensional axis space.

Persona artifacts (axes, descriptors, cards) can be saved and replayed for reproducibility.

## Judge system

Judge personas are reused across questions within a benchmark for comparability:

| Benchmark | Judge families |
|-----------|---------------|
| AIME25 | `math` |
| GPQA | `biology`, `chemistry`, `physics` |
| HLE-Verified | `math`, `physical_sciences`, `medicine`, `computer_science`, `humanities`, `applied_professional_reasoning` |

Each question is routed to the appropriate judge family based on its domain. Judge cards are persisted in a bank directory so the same judge evaluates all questions of a given type.

## Supported datasets

- **AIME25** -- competition math (integer answers in `\boxed{}`)
- **GPQA** -- graduate-level science, multiple choice A/B/C/D (biology, chemistry, physics)
- **HLE-Verified** -- broad expert-level questions (math, physical sciences, medicine, CS, humanities, applied reasoning). Variants: `verified`, `revised`, `verified_full`.

## Modes

| Mode | Description |
|------|-------------|
| `single` | One generation per question |
| `majority` | Sample multiple generations, pick the mode answer |
| `debate` | Multi-agent, multi-round debate with judge |
| `personas` | Generate and persist persona artifacts only (no inference) |

Default: `single,debate` (comma-separated).

## Repository layout

```
src/debate_v_majority/
├── cli/                 CLI entry, orchestration, output formatting
│   ├── args.py          Argument parser
│   ├── main_impl.py     Main CLI implementation
│   ├── debate_runner.py Multi-agent debate loop
│   ├── sample_runner.py Single/majority mode
│   ├── judge.py         Judge prompt construction and parsing
│   └── ...
├── engines/             Inference engine abstractions
│   ├── gemini_api.py    Gemini API client (google-genai SDK)
│   └── ...
├── datasets/            Dataset adapters (AIME25, GPQA, HLE)
├── personas/            Persona generation, axes, sampling, judge bank
│   ├── generator.py     Descriptor and card generation
│   ├── judge_bank.py    Benchmark-family judge card bank
│   ├── axes.py          Fixed and task-specific diversity axes
│   ├── sampling.py      Maximin and Halton sampling
│   └── ...
├── shared/              Answer parsing, normalization, token counting
└── tools/               Analysis and trace rendering utilities
tests/                   Unit tests
```

## Setup

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
# or
export GOOGLE_API_KEY="your-key-here"
```

If a local `.env` file contains `GEMINI_API_KEY=...`, that value takes precedence over both the constructor `api_key` argument and OS environment variables.

## Usage

```bash
# single + debate on 5 AIME25 questions
debate-v-majority \
  --dataset aime25 \
  --mode single,debate \
  --subset_n 5 \
  --model_name gemini-3-flash-preview

# debate with personas on GPQA
debate-v-majority \
  --dataset gpqa \
  --mode debate \
  --use_personas \
  --n_agents 5 \
  --n_rounds 3 \
  --persona_save_artifacts \
  --persona_artifacts_dir artifacts/ \
  --judge_bank_dir judge_bank/ \
  --model_name gemini-3-flash-preview

# majority vote on HLE-Verified
debate-v-majority \
  --dataset hle \
  --mode majority \
  --majority_samples 5 \
  --model_name gemini-3-flash-preview

# generate persona artifacts only (no inference)
debate-v-majority \
  --dataset gpqa \
  --mode personas \
  --persona_save_artifacts \
  --persona_artifacts_dir artifacts/
```

## Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `gemini-3-flash-preview` | Model for debaters |
| `--dataset` | `aime25` | `aime25`, `gpqa`, or `hle` |
| `--mode` | `single,debate` | Comma-separated modes to run |
| `--n_agents` | `5` | Number of debate agents |
| `--n_rounds` | `3` | Debate update rounds (after initial solve) |
| `--use_personas` | off | Enable persona-conditioned agents |
| `--persona_backend` | `auto` | `heuristic`, `llm`, or `auto` |
| `--judge_persona_mode` | `benchmark_family_bank` | How judge cards are assigned |
| `--judge_trace_mode` | `visible_plus_thought_summary` | What the judge sees from agents |
| `--subset_n` | `20` | Number of items to sample |
| `--max_run_cost_usd` | none | Cost cap per run |
| `--out_dir` | auto | Output directory |
| `--emit_trace_level` | `full` | `minimal` or `full` trace in output |

## Utilities

```bash
# aggregate analysis across runs
analyze-results --results-dir results/ --out-dir _autogen/

# render a run row as readable text
trace2txt --input results/debate_gpqa_*.jsonl --item "gpqa:some-item-id"
```

## Testing

```bash
pytest -q
```

## Engine

Uses the Gemini API via `google-genai >= 1.66.0`. Thinking mode is handled natively through the API's `part.thought` attribute -- thought summaries are requested via `ThinkingConfig(include_thoughts=True)` and separated from visible output at the response level.
