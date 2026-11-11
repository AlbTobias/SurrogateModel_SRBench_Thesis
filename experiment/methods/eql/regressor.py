#!/usr/bin/env python3
"""SRBench EQL adapter used by the surrogate benchmark.

The base estimator, candidate grid, and equation-export helper originate from
the upstream SRBench EQL submission.  The ``GridSearchCV`` construction remains
commented out, so the pinned image uses the direct ``EQL`` instance assigned to
``est``; the candidate grid is retained only as upstream adapter context and is
not searched by this project.  ``experiment.evaluate_surrogate`` clones this
estimator and applies the fixed parameters from the selected versioned
benchmark configuration.
"""

from eql.est import EQL
import pandas as pd
from sklearn.model_selection import GridSearchCV


base = EQL(n_iter=10_000)

reg = (1e-4, 1e-3, 1e-2, 5e-2)
n_layers = (1, 2)
functions = [
    "id;mul;cos;sin;exp;square;sqrt;id;mul;cos;sin;exp;square;sqrt;log",
    "id;mul;cos;div;sqrt;cos;sin;div;mul;mul;cos;id;log"
]

hyper_params = []

for r, l in zip(reg, n_layers):
    for f in functions:
        hyper_params.append({
                'reg':[r],
                'n_layers':[l],
                'functions':[f]
                })
        
# Inactive upstream reference only: the thesis benchmark does not construct or
# execute this GridSearchCV search.
# est = GridSearchCV(estimator=base, param_grid=hyper_params, cv=2, refit=True, n_jobs=4, verbose=0)
est = base

def model(est, X=None):
    model_str = str(est.get_eqn())

    if isinstance(X, pd.DataFrame):
        for i,f in reversed(list(enumerate(X.columns))):
            model_str = model_str.replace(f'x{i}',f)

    return model_str


eval_kwargs = {
    # Optional upstream smoke-test parameters remain intentionally inactive.
    # 'test_params': {'n_iter': [10], **hyper_params[-1]},
    'use_dataframe':False
}
