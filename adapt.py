"""
Main function for few-shot & semi-supervised adaptation.
"""

import argparse
import torch
import os
import time
import json

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from datetime import datetime

from documents.local_data.constants import PATH_FEATURES, PATH_RESULTS

from data.configs.tasks_batch import * 
from data.configs.tasks_configs import get_task_cfg
from data.split.realistic import split

from libs.archs.lp import LinearSoftmax
from libs.archs.lpp2 import LPpp
from libs.archs.tipadapt import TIPAd
from libs.metrics.utils import evaluate
from libs.solvers import runner

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # Set dict to gather all results
    results = {}
    results["average"] = {"aca": {}}

    # Set dict to gather results per task
    for task in args.tasks:
        results[task["name"]] = {"aca": {}}

    # Run over shots
    for K in args.shots_l:
        # Run over tasks
        for task in args.tasks:
            
            # Retrieve task splits (train, test).
            splits_cfg = get_task_cfg(task["name"])

            # Load pre-computed feature embeddings and labels for train and test
            data_adapt = np.load(PATH_FEATURES + splits_cfg["train"] + "_" + task["vlm"] + ".npz", allow_pickle=True)
            features_adapt, labels_adapt = data_adapt['feats_ds'], data_adapt['refs_ds']
            data_test = np.load(PATH_FEATURES + splits_cfg["test"] + "_" + task["vlm"] + ".npz", allow_pickle=True)
            features_test, labels_test = data_test['feats_ds'], data_test['refs_ds']

            # Load pre-computed class prototypes using the VLM's text encoder and class descriptions
            textual_prototypes = data_adapt["textual_prototypes"]
            logit_scale = data_adapt["logit_scale"]

            # Number of categories
            C = len(np.unique(labels_test))

            # Number of annotated samples.
            N = K * C

            # Label-marginal distribution for sampling
            label_marginal = np.bincount(np.int32(labels_test)) / len(labels_test)

            # Number of unlabeled samples to use - to ensure a homogeneous label-distribution in all splits.
            freqs_adapt = np.bincount(np.int32(labels_adapt))
            available_u_samples = np.round(freqs_adapt - args.shots_l[-1]*label_marginal*C)
            M = int(np.min(available_u_samples / (label_marginal*C)))*C
            M = np.min([M, C*args.shots_u])

            # Compute shillouette scopre and zero-shot predictions
            if K==0:
                
                # Compute and print shillouette scores
                if args.dataset_silhouette:
                    from sklearn.metrics import silhouette_samples
                    silh_s = silhouette_samples(features_test, labels_test, metric='cosine')
                    silh = np.mean([np.mean(silh_s[labels_test==c]) for c in np.unique(labels_test)])
                    print("{:10s} - Silhouette score={:.4f}".format(task["name"], np.round(silh, 4)))

                # Set zero-shot weights for linear classifier
                model = LinearSoftmax(D=features_adapt.shape[-1], C=C, norm=True, weights=textual_prototypes, logit_scale=logit_scale)
                # Compute scores.
                with torch.no_grad():
                    scores_test = model(torch.tensor(features_test)).cpu().numpy()
                # Evaluate
                _, aca = evaluate(scores_test, labels_test)
                # Add task results
                results[task["name"]]["aca"][K] = aca
                continue

            # Run experiments over random seeds
            aca_seeds = []
            for seed in tqdm(range(args.seeds), leave=True, desc="K={:2d} @ {:10s}".format(K, task["name"])):

                # Set classification head
                if args.solver in ["SS-Text-U", "SS-Text", "SS-Text+", "SS", "LP", "CLAP"]:
                    model = LinearSoftmax(D=features_adapt.shape[-1], C=C, norm=True, weights=textual_prototypes, logit_scale=logit_scale)
                elif args.solver in ["LP++"]:
                    model = LPpp(textual_prototypes)
                elif args.solver in ["TIPAd"]:
                    model = TIPAd(textual_prototypes)
                else:
                    print("Solver not implemented.")
                    break

                # Sampling an adaptation subset with some annotated shots.
                features_a, labels_a, features_u, labels_u = split(
                    features_adapt, labels_adapt, N=N, p=label_marginal, seed=seed)

                # Sample the maximum unlabeled subset s.t. the label-marginal is the one of the test distribution.
                features_u, labels_u, _, _ = split(features_u, labels_u, N=M, p=label_marginal, seed=seed)

                # Fit linear probe.
                model = runner.run_adaptation(features_a, labels_a, model, classes=C, solver=args.solver, features_u=features_u)

                # Compute scores.
                with torch.no_grad():
                    scores_test = model(torch.tensor(features_test)).cpu().numpy()

                # Evaluate
                _, aca = evaluate(scores_test, labels_test)

                # Save coverage and set size.
                aca_seeds.append(aca)

            # Add task results
            results[task["name"]]["aca"][K] = np.round(np.mean(aca_seeds), 2).item()

        # Average result across tasks
        results["average"]["aca"][K] = np.round(np.mean([results[task["name"]]["aca"][K] for task in args.tasks]), 2).item()
    
    # Print results
    print("Average performance for all datasets:")
    print(results["average"])
    if not os.path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)
    with open(PATH_RESULTS + "{solver}_{exp_id}_{time}".format(
                solver=args.solver, exp_id=args.exp_id, time=datetime.now().strftime("%m%d%H%M")), "w") as file: 
            json.dump(results, file, indent=1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', default='main')

    # Tasks.
    parser.add_argument('--tasks', default=TASKS)

    # Solver for adaptation.
    parser.add_argument('--solver', default='SS-Text-U', help='Few-shot solver',
                        choices=["SS-Text-U", "SS-Text", "SS-Text+", "SS", "LP", "CLAP", "LP++", "TIPAd"])

    # Few-shot sampling data.
    parser.add_argument('--shots_l', default=[0, 1, 2, 4, 8, 16], help='Number of shots for adaptation per class', type=list)
    parser.add_argument('--shots_u', default=24, help='Number of shots for adaptation per class', type=int)

    # Number of seeds.
    parser.add_argument('--seeds', default=50, type=int, help='Number of experiments repetitions')

    # Other options.
    parser.add_argument('--dataset_silhouette', default=False, type=bool)

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()