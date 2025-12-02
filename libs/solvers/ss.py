import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ss_solver(features, labels, model, classes):

    features = torch.tensor(features).float().to(device)
    labels = torch.tensor(labels).long().to(device)

    # Compute new class centers
    with torch.no_grad():

        # Labels to ohe
        ohe = torch.nn.functional.one_hot(labels, num_classes=classes).float()

        # Compute new class centers (visual)
        vision_mu = torch.einsum('ij,ik->jk', ohe, features)

        # Normalize
        vision_mu /= ohe.sum(0).unsqueeze(-1)

        # Check prototypes for categories with missing examples - and modify its prototype with a random vector.
        idx = torch.where(torch.isnan(vision_mu)[:, 0])
        if len(idx) > 0:
            for i in idx:
                vision_mu[i, :] = torch.nn.init.kaiming_normal_(torch.empty([1, vision_mu.shape[-1]])).to(device)

    # Set Adapter
    model.prototypes.data = vision_mu.to(device)

    return model