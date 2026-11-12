"""Strict within-one-metric task counts. Never merge endpoints into success."""
import math

METRICS=('c','ibs','mae_uncensored','c_from_point')

def metric_leads(table,datasets,method,comparators,metric):
    if metric not in METRICS:raise ValueError('Unknown endpoint')
    comparators=[m for m in comparators if m!=method]
    if not comparators:raise ValueError('A comparison needs other methods')
    higher=metric in {'c','c_from_point'};tasks={}
    for dataset in datasets:
        eligible=all(table[dataset][m]['complete_search'] for m in [method]+comparators)
        if eligible:
            own=table[dataset][method]['metrics'][metric]['mean']
            values=[table[dataset][m]['metrics'][metric]['mean'] for m in comparators]
            if not all(math.isfinite(v) for v in [own]+values):raise ValueError('Nonfinite complete result')
            best=(max if higher else min)(values)
            first=own>best if higher else own<best
            margin=own-best if higher else best-own
        else:own=best=margin=None;first=False
        tasks[dataset]=dict(eligible=eligible,first=bool(first),mean=own,
                            best_comparator_mean=best,favorable_margin=margin)
    return dict(count=sum(v['first'] for v in tasks.values()),tasks=tasks)

def primary_target_met(wins,protocol):
    if protocol['primary_metric']!='c':raise ValueError('This campaign preregisters C as its primary metric')
    return wins[protocol['primary_candidate']]['external26']['c']['count']>=protocol['single_metric_threshold']
