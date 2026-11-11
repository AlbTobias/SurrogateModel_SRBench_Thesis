"""SRBench PySR adapter used by the surrogate benchmark.

The estimator definition, candidate parameter sets, and export helpers derive
from the upstream SRBench PySR submission.  The pinned image uses the direct
``PySRRegressor`` instance defined below and does not require a separate local
compatibility branch.  The later ``hyper_params = [{}]`` assignment preserves
the upstream adapter's disabled tuning interface; it is not a search performed
by this project.  During benchmark execution, ``experiment.evaluate_surrogate``
clones ``est`` and applies the fixed parameters from the selected versioned
benchmark configuration.
"""

from pysr import PySRRegressor
from multiprocessing import cpu_count

hyper_params = [
    {
        'niterations' : [1_000_000_000],
        'maxsize' : [20],
        'maxdepth' : [20],
    },
    {
        'niterations' : [1_000_000],
        'maxsize' : [30],
        'maxdepth' : [20],
    },
    {
        'niterations' : [1_000_000],
        'maxsize' : [50],
        'maxdepth' : [20],
    },
    {
        'niterations' : [1_000_000],
        'maxsize' : [30],
        'maxdepth' : [20],
        'population_size' : [200],
    },
]

def complexity(est):
    return int(est.get_best()["complexity"])


def model(est, X=None):
    return str(est.sympy())


est = PySRRegressor(
    niterations=1_000_000_000,
    ncyclesperiteration=2_500,
    population_size=100,
    populations=2, # originally max(15, 2*n_cpu), but srbench runs on a single core
    verbosity=0,
    # Upstream adapter time allowance retained as context. The final benchmark
    # replaces it with the timeout in the versioned configuration.
    timeout_in_seconds=60*60 - 10*60,
    maxsize=30,
    maxdepth=20,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "exp", "log", "sqrt"],
    constraints={
        **dict(
            sin=9,
            exp=9,
            log=9,
            sqrt=9,
        ),
        **{"/": (-1, 9)}
    },
    nested_constraints=dict(
        sin=dict(
            sin=0,
            exp=1,
            log=1,
            sqrt=1,
        ),
        exp=dict(
            exp=0,
            log=0,
        ),
        log=dict(
            exp=0,
            log=0,
        ),
        sqrt=dict(
            sqrt=0,
        )
    ),
    # Upstream single-process setting; the final configuration also enforces
    # deterministic, single-process execution explicitly.
    procs=1, # cpu_count(),
    multithreading=False,
    batching=True,
    batch_size=50,
    turbo=True,
    weight_optimize=0.001,
    adaptive_parsimony_scaling=1_000.0,
    parsimony=0.0,
)

# Disable the upstream tuning grid. The project uses fixed configuration values
# and performs no adapter-level hyperparameter search.
hyper_params = [{}]

eval_kwargs = {
    "test_params": dict(
        niterations=3,
        ncyclesperiteration=500,
        populations=3,
    )
}
