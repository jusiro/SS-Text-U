import torch
import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ss_text_solver(features, labels, model, classes, text_lambda="adaptive_shots",
                   balanced_importance=False, repel=False):

    # %---------------------------------------------------------------
    # Init, constants.
    features = torch.tensor(features).float()
    labels = torch.tensor(labels).long()
    textual_prototypes = model.prototypes.clone()  # Extract the zero-shot weights from init head.

    # Set temperature scaling parameter, tau.
    tau = (1 / model.logit_scale.exp())

    # Number of samples
    N = labels.shape[0]

    # Compute new class centers
    with torch.no_grad():

        # Labels to ohe
        ohe = torch.nn.functional.one_hot(labels, num_classes=classes).float()

        # Compute new class centers (visual)
        vision_mu = torch.einsum('ij,ik->jk', ohe, features)

        if text_lambda == "adaptive_shots":
            if balanced_importance:
                lambda_text = torch.tensor((1 / (N)))
            else:
                lambda_text = (1 / torch.sum(ohe, dim=0, keepdim=True)).t()
        elif text_lambda == "adaptive_perf":  # CLAP - alike constraint

            # Estimate zero-shot performance
            z = model(features.to(device))

            # Get one-hot encoding ground-truth
            labels_one_hot = ohe.clone()

            # Compute prior
            lambda_text = torch.diag(labels_one_hot.t().to(device) @ z.to(device))
            lambda_text /= labels_one_hot.sum(dim=0).to(device)
            lambda_text = lambda_text.clone().cpu()

            #  Correct Nans
            lambda_text[torch.isnan(lambda_text)] = torch.mean(
                lambda_text[torch.logical_not(torch.isnan(lambda_text))])

        else:  # Text lambda fixed
            lambda_text = torch.tensor(0.1)

        # Compute optimum weights via ss-text solver
        new_mu = vision_mu * (1 / N) * (1 / lambda_text) * (1 / tau) + \
                 textual_prototypes

        if repel:
            delta = (2 * new_mu - new_mu.mean(0))
            delta /= delta.norm(dim=-1, keepdim=True)
            new_mu = new_mu - delta

    # Set Adapter
    model.prototypes.data = new_mu.to(device)

    return model