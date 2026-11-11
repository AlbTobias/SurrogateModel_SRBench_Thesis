# Symbolic Regression for Surrogate Modeling

This repository contains the benchmark implementation and result artefacts for
the bachelor's thesis *Symbolic Regression for Surrogate Modeling: A Systematic
Study*. It evaluates six open-source symbolic-regression frameworks on fifteen
analytical, simulator-generated, experimental, and measured surrogate-modeling
problems.

The project adapts the common estimator interface and published container images
from [SRBench](https://github.com/cavalab/srbench). It adds a fixed surrogate
problem collection, deterministic dataset construction, two input-scaling
conditions, repeated fitting trials, framework-independent symbolic analysis,
and an auditable result-resolution procedure.

## Benchmark scope

The final experiment contains:

- 6 symbolic-regression frameworks;
- 15 surrogate-modeling problems;
- 2 input-scaling conditions;
- 10 repetition labels per configuration;
- 1,800 planned trials in total.

Of these trials, 1,798 completed successfully and two are retained as explicit
failures.

### Frameworks

| Framework | Principal approach |
|---|---|
| gplearn | Conventional tree-based genetic programming |
| Operon | Performance-oriented tree-based genetic programming |
| PySR | Multi-population evolutionary symbolic regression |
| GeneticEngine | Grammar-guided genetic programming |
| ITEA | Interaction--transformation evolutionary search |
| EQL | Differentiable equation-learning network |

Every framework runs through a pinned SRBench container image [`containers/images.lock`](containers/images.lock).

### Problems

| Problem | Type | Inputs | Train/test observations |
|---|---|---:|---:|
| Cantilever | Analytical structural model | 5 | 400/4,000 |
| Borehole | Analytical hydrological model | 8 | 400/4,000 |
| Piston | Analytical thermodynamic model | 7 | 400/4,000 |
| Combined Cycle Power Plant | Measured energy-system data | 4 | 400/4,000 |
| Naval Propulsion | Simulator-generated propulsion data | 3 | 400/4,000 |
| Wing Weight | Analytical aircraft-design model | 10 | 400/4,000 |
| Gas Turbine NOx | Measured emissions data | 9 | 400/4,000 |
| Concrete Strength | Experimental materials data | 8 | 400/592 |
| Energy Efficiency | Simulator-generated building data | 8 | 400/368 |
| Airfoil Self-Noise | Experimental aeroacoustic data | 5 | 400/1,103 |
| OTL Circuit | Analytical electronic-circuit model | 6 | 400/4,000 |
| Robot Arm | Analytical robotics model | 8 | 400/4,000 |
| Yacht Hydrodynamics | Experimental hydrodynamic data | 6 | 210/98 |
| Electrical Grid Stability | Simulator-generated grid data | 11 | 400/4,000 |
| Lattice Physics | Simulator-generated reactor-physics data | 39 | 400/4,000 |

Equations, feature definitions, domain bounds, data sources, licenses, hashes,
and split procedures are documented in
[`docs/surrogate-problems.md`](docs/surrogate-problems.md) and
[`data/README.md`](data/README.md). Downloaded and generated data tables are not
committed; the preparation scripts recreate them locally.

## Evaluation protocol

Every problem is evaluated with raw inputs and with fixed domain normalization:

```text
z = 2 * (x - lower) / (upper - lower) - 1
```

The lower and upper bounds are defined by the problem specification rather than
estimated from training or test observations. Targets remain in their original
units.

The final repetition labels are:

```text
2, 3, 5, 7, 11, 13, 17, 19, 23, 29
```

They control the framework seed where supported but never change the fixed
training or reference-test tables. The employed ITEA adapter does not expose a
supported seed parameter, so its executions are reported as uncontrolled
repetitions rather than seed-controlled trials.

The benchmark records:

- MAE, RMSE, range-normalized RMSE, coefficient of determination, and maximum
  absolute error on the held-out reference-test set;
- fitting, repeated prediction, and expression-extraction time;
- expression availability and parsing coverage;
- expression-tree node count, depth, operators, constants, and variables;
- corresponding structural measures after bounded symbolic simplification;
- explicit simplification fallbacks, execution failures, and missing trials.

Predictive metrics are computed from the framework estimator's predictions.
Expression extraction is the separate adapter-specific export of the fitted
model. SymPy parsing, transformation to physical coordinates, simplification,
and structural measurement are performed afterward and do not contribute to
the reported algorithm runtime.

The complete measurement and post-processing policy is documented in
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md).

## Requirements

The benchmark was developed and executed on Linux. Running trials requires:

- Python 3.11 for host-side coordination and analysis;
- Docker Engine to run containers;

Create the host environment with:

```bash
python3 -m venv --upgrade-deps .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Confirm that Docker is available before starting an experiment:

```bash
docker version
```

The algorithm containers are CPU-only in this workflow. Shared execution
controls request one thread per trial.

## Running the benchmark

### One trial

The trial runner accepts an algorithm, repetition label, problem, and scaling
condition:

```bash
BENCHMARK_CONFIG=configs/benchmark_suite_v11.json \
./scripts/run_benchmark_trial.sh gplearn 2 cantilever raw
```

Valid scaling conditions are `raw` and `domain_minmax`.

### One complete problem

This command runs all six frameworks, both scaling conditions, and all ten
repetitions for one problem. It then performs symbolic analysis and writes the
per-problem summaries:

```bash
BENCHMARK_CONFIG=configs/benchmark_suite_v11.json \
BENCHMARK_PROBLEMS=cantilever \
./scripts/run_benchmark_group.sh
```

Replace `cantilever` with any problem identifier from
[`configs/benchmark_suite_v11.json`](configs/benchmark_suite_v11.json).

The group runner can also restrict algorithms, repetitions, or scaling
conditions:

```bash
BENCHMARK_ALGORITHMS="operon pysr" \
BENCHMARK_SEEDS="2 3" \
BENCHMARK_SCALINGS=domain_minmax \
BENCHMARK_PROBLEMS=borehole \
./scripts/run_benchmark_group.sh
```

### Complete matrix

Running the group script without restrictions executes the complete 1,800-trial
matrix and may require substantial time:

```bash
BENCHMARK_CONFIG=configs/benchmark_suite_v11.json \
./scripts/run_benchmark_group.sh
```

## Symbolic post-processing

Symbolic analysis is stored separately from each immutable trial record. To
resume or repeat analysis without fitting a model again, run:

```bash
.venv/bin/python scripts/analyze_benchmark_results.py \
  --problem borehole \
  --benchmark benchmark_suite_v11 \
  --config configs/benchmark_suite_v11.json \
  --timeout 60 \
  --memory-limit-mb 2048 \
  --jobs 1
```

Each successful analysis writes a `seed-N.analysis.json` sidecar containing the
SHA-256 hash of its source trial. Stale sidecars are rejected. If bounded
simplification exceeds its time or memory limit, the pipeline retries without
simplification and retains the parsed unsimplified expression as a documented
fallback.

## Resolving and plotting the reported results

The problem collection was extended incrementally. The thesis result matrix
therefore resolves each problem--scaling--framework--repetition combination to
its newest result from a protocol-compatible suite between v7 and v11. The
resolver verifies configuration compatibility and analysis-sidecar hashes before
aggregation.

Regenerate the final manifest and summary with:

```bash
.venv/bin/python scripts/summarize_latest_results.py \
  --config configs/benchmark_suite_v11.json
```

This writes:

- `results/latest/manifest.csv`, containing the selected source record for every
  planned trial;
- `results/latest/summary.csv`, containing the aggregated framework--problem--
  scaling results.

Regenerate all thirteen result figures with:

```bash
.venv/bin/python scripts/plot_benchmark_results.py \
  --config configs/benchmark_suite_v11.json
```

The plotting command stops if any expected trial is missing or if a selected
successful trial lacks a current analysis sidecar.

## Result layout

```text
results/
├── <problem>/<suite>/<scaling>/<algorithm>/seed-N.json
├── <problem>/<suite>/<scaling>/<algorithm>/seed-N.analysis.json
├── <problem>/<suite>/<scaling>/failures/<algorithm>/seed-N.json
├── latest/manifest.csv
├── latest/summary.csv
└── figures/
```

Trial JSON files contain dataset hashes, applied configuration, environment
metadata, predictive metrics, timings, and the extracted expression. Analysis
sidecars contain the separate parsing, coordinate-transformation,
simplification, and structural measurements.

Intermediate framework runtime files are deliberately not part of the
publication repository.

## Tests

Run the focused tests for dataset construction, scaling, prediction-shape
handling, symbolic analysis, and sidecar validation with:

```bash
python -m pytest -q \
  surrogate/test_problems.py \
  experiment/metrics/test_analysis_postprocessing.py \
  experiment/metrics/test_expression_analysis.py \
  experiment/metrics/test_input_scaling.py \
  experiment/metrics/test_prediction_shape.py \
  scripts/test_analyze_benchmark_results.py
```

## Reproducibility boundaries

- Framework configurations are fixed across problems and repetitions.
- No problem-specific hyperparameter optimization is performed.
- Search budgets are algorithm-specific and do not represent equal compute.
- Runtime comparisons describe the evaluated configurations on the documented
  execution environment, not hardware-normalized framework limits.
- Dataset-construction seeds are independent of repetition labels.
- External source files and generated tables are verified using recorded hashes.
- Failures and unavailable symbolic analyses remain explicit.
- Structural complexity is a syntactic indicator and does not by itself prove
  scientific meaning or physical plausibility.

## Repository structure

| Path | Purpose |
|---|---|
| `algorithms/` | Environment and installation metadata for the six frameworks |
| `configs/` | Versioned benchmark configurations used by the final resolver |
| `containers/` | Immutable SRBench container-image references |
| `data/README.md` | Dataset schemas, equations, sources, and split details |
| `docs/` | Problem and evaluation-protocol documentation |
| `experiment/` | Common evaluator, framework adapters, and symbolic metrics |
| `surrogate/` | Analytical generators and external-data preparation scripts |
| `scripts/` | Trial execution, post-processing, aggregation, and plotting |
| `results/` | Resolved trial artefacts, summaries, and figures |

## Upstream attribution and license

This work is derived in part from SRBench. The imported upstream revision is
recorded in [`SRBENCH_UPSTREAM_COMMIT`](SRBENCH_UPSTREAM_COMMIT), and the
framework images are identified in
[`containers/images.lock`](containers/images.lock)