"""SRBench ITEA adapter used by the surrogate benchmark.

This module retains the upstream SRBench pyITEA estimator, candidate parameter
sets, expression conversion, and environment compatibility setup.  The pinned
image uses the direct ``ITEARegressor`` instance defined below.  The retained
``pre_train`` helper belongs to the upstream evaluator interface and is not
called by this project's ``experiment.evaluate_surrogate`` workflow.  For the
final runs, the evaluator clones ``est`` and applies the fixed parameters from
the selected versioned benchmark configuration.
"""

import sys
import os
os.environ["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + "/lib"
import pyITEA as itea
from itertools import product

hyper_params = [
    {
        'exponents' : ((0,5),),
        'termlimit' : ((2,5),),
        'transfunctions' : ('[Id, Sin]',),
        'npop' : (1000,),
        'ngens' : (1000,),
    },
    {
        'exponents' : ((0,5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Sin]',),
        'npop' : (1000,),
        'ngens' : (1000,),
        'nonzeroexps' : (10,),
    },
    {
        'exponents' : ((-5,5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Sin]',),
        'npop' : (1000,),
        'ngens' : (1000,),
    },
    {
        'exponents' : ((-5, 5),),
        'termlimit' : ((2, 5),),
        'transfunctions' : ('[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]',),
        'npop' : (1000,),
        'ngens' : (1000,),
    },
    {
        'exponents' : ((0, 5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]',),
        'npop' : (1000,),
        'ngens' : (1000,),
    },
    {
        'exponents' : ((-5, 5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]',),
        'npop' : (1000,),
        'ngens' : (1000,),
        'nonzeroexps' : (10,),
    },
]


def pre_train(est, X, y):
    """Legacy SRBench hook for reducing generations on very large datasets."""
    
    # Retained for upstream compatibility; the thesis evaluator does not call
    # this hook.
    if len(X) >= 10_000:
        est.ngens = 250
    print('ITEA ngens adjusted to', est.gens)


# Direct estimator used by the pinned image. The versioned benchmark
# configuration overrides its final population and generation settings.
eval_kwargs = {'scale_x': False, 'scale_y': False}
est = itea.ITEARegressor(npop=1000, ngens=500, exponents=(-1, 1), termlimit=(2,
    2), nonzeroexps=1, 
    transfunctions= '[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]'
    )

def complexity(e):
    return e.len

def model(e, X):
    # Convert pyITEA's expression syntax to the notation accepted by SymPy.
    new_model = e.sympy.replace("^","**")
    
    new_model = new_model.replace('sqrtAbs','PSQRT')

    for i,f in reversed(list(enumerate(X.columns))):
        new_model = new_model.replace(f'x{i}',f)
        
    return new_model
