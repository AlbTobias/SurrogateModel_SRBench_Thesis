"""SRBench gplearn adapter used by the surrogate benchmark.

The estimator definition and adapter functions originate from the SRBench
gplearn submission.  This project uses the adapter's direct
``SymbolicRegressor`` instance; no image-specific compatibility branch is
required.  For benchmark runs, ``experiment.evaluate_surrogate`` clones this
instance and applies the parameters from the selected versioned benchmark
configuration.  The defaults and ``hyper_params`` below are therefore retained
as upstream adapter context rather than as the final thesis configuration.
"""

from gplearn.genetic import SymbolicRegressor
import re

hyper_params = []
for p, g in zip([1000,500,100],[500,1000,5000]):
    for fs in [('add', 'sub', 'mul', 'div', 'log','sqrt'),
               ('add', 'sub', 'mul', 'div', 'log','sqrt', 'sin','cos')
              ]:
        hyper_params.append({
            'population_size' : [p],
            'generations' : [g],
            'function_set': [fs]
            })

est = SymbolicRegressor(
                        tournament_size=20,
                        init_depth=(2, 6),
                        init_method='half and half',
                        metric='mean absolute error',
                        parsimony_coefficient=0.001,
                        p_crossover=0.9,
                        p_subtree_mutation=0.01, 
                        p_hoist_mutation=0.01, 
                        p_point_mutation=0.01, 
                        p_point_replace=0.05,
                        max_samples=1.0,
                        function_set= ('add', 'sub', 'mul', 'div', 'log',
                                        'sqrt', 'sin','cos'),
                        population_size=1000,
                        generations=500
                       )

# Used only when SRBench's evaluator is called with test=True. These values keep
# the interface smoke test short without changing the benchmark configuration
# above.
eval_kwargs = {
    'test_params': {
        'population_size': 100,
        'generations': 5,
    },
    'max_train_samples': 250,
}


def model(est, X=None):
    return str(est._program)

def complexity(est):
    # Adapter-specific size estimate based on tokens in gplearn's program
    # string. The thesis comparison uses the framework-independent SymPy
    # node, depth, operator, constant, and variable measures instead.
    return len(re.split(r'\(|,',model(est)))
