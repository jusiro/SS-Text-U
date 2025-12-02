import torch

from .gd import gd_solver
from .ss import ss_solver
from .lpp2 import lp_pp_solver
from .sstext import ss_text_solver
from .sstextu import sstext_u_solver
from .tipadapt import tipadapt_solver

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_adaptation(features, labels, model, classes, solver="SS-Text+", samples_text=None, labels_text=None,
                   features_u=None, true_label_marginal=None):

    # Extended cross-modal dataset
    if solver == "CrossModal":
        if samples_text is not None:
            features = torch.concatenate([features, samples_text], dim=0)
            labels = torch.concatenate([labels, labels_text], dim=0)

    if solver in ["LP", "LPcw", "LPldam", "TaskRes", "ClipAdapt", "CrossModal", "CLAP"]:
        model = gd_solver(features, labels, model, classes=classes,
                          weights=("cw" in solver or "ldam" in solver),
                          ldam=("ldam" in solver),
                          clap=(solver == "CLAP"))
    elif solver == "SS":
        model = ss_solver(features, labels, model, classes=classes)
    elif solver == "SS-Text":
        model = ss_text_solver(features, labels, model, classes=classes,
                                  balanced_importance=True, repel=False)
    elif solver == "SS-Text+":
        model = ss_text_solver(features, labels, model, classes=classes,
                                  balanced_importance=False, repel=True)
    elif solver == "SS-Text-U":
        model = sstext_u_solver(features, labels, model, features_u, classes=classes,
                                num_iters_mm=3, num_iters_ot=10, ratio_label_marginal_correction=4,
                                true_label_marginal=true_label_marginal)
    elif solver == "LP++":
        model = lp_pp_solver(features, labels, model, classes=classes)
    elif solver == "LP++(TF)":
        model = lp_pp_solver(features, labels, model, classes=classes, epochs=0)
    elif solver == "TIPAd":
        model = tipadapt_solver(features, labels, model, classes=classes)
    elif solver == "TIPAdFT":
        model = tipadapt_solver(features, labels, model, classes=classes)
        model = gd_solver(features, labels, model, classes=classes)
    else:
        print("Solver not implemented...")
        return model

    return model
