import argparse
import pathlib
import warnings
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_evaluations(logdir: pathlib.Path, regex: str) -> List[Dict[str, np.ndarray]]:
    """ look for evaluations.npz in the subdirectories and return is content """
    evaluations = []
    for eval_file in logdir.glob(f"{regex}/evaluations.npz"):
        data = np.load(eval_file)
        evaluations.append(dict(data))
    if len(evaluations) == 0:
        warnings.warn(f"cannot find any file for `{logdir}/{regex}/evaluations.npz`", UserWarning)
    return evaluations


def aggregate_evaluations(evaluations: List[Dict[str, np.ndarray]], params: Dict) -> Dict[str, np.ndarray]:
    """ aggregate data in the list to produce overall figure """
    # make flat collection of x,y data
    xx, yy = [], []
    for evaluation in evaluations:
        assert params['x'] in evaluation, f"{params['x']} is not in evaluation keys ({evaluation.keys()})"
        assert params['y'] in evaluation, f"{params['y']} is not in evaluation keys ({evaluation.keys()})"
        xx.append(evaluation[params['x']])
        yy.append(np.mean(evaluation[params['y']], axis=-1))  # eg, average results over multiple eval episodes
    xx = np.concatenate(xx, axis=0)
    yy = np.concatenate(yy, axis=0)
    assert xx.shape == yy.shape, f"xx, yy dimensions don't match (xx shape:{xx.shape}, yy shape:{yy.shape}"
    # aggregate
    df = pd.DataFrame({'x': xx, 'y': yy})
    bins = np.arange(0, max(xx) + params['binning'], params['binning'])
    df['xbin'] = pd.cut(df['x'], bins=bins, labels=bins[:-1])
    aggregated = df.groupby("xbin").agg(['mean', 'std'])['y'].reset_index()
    return {'x': aggregated['xbin'].values,
            'mean': aggregated['mean'].values,
            'std': aggregated['std'].values}


def plot_data(data: Dict[str, np.ndarray], ax: plt.Axes, **kwargs):
    assert all([key in data for key in ['x', 'mean', 'std']]), f'x, mean, std not found in data (keys: {data.keys()})'
    ax.plot(data['x'], data['mean'], **kwargs)
    ax.fill_between(data['x'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.5)


def main(args):
    # prepare plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # plot data
    for regex in args.regex:
        evaluations = get_evaluations(args.logdir, regex)
        if not evaluations:
            continue
        data = aggregate_evaluations(evaluations, params={'x': 'timesteps', 'y': 'results', 'binning': args.binning})
        plot_data(data, ax, label=regex)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=pathlib.Path, required=True)
    parser.add_argument("--regex", type=str, default="**", nargs="+",
                        help="for each regex, group data for `{logdir}/{regex}/evaluations.npz`")
    parser.add_argument("--binning", type=int, default=15000)
    args = parser.parse_args()
    main(args)
