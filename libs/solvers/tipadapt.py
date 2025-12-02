import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def tipadapt_solver(features, labels, model, classes):

    features = torch.tensor(features).float().to(device)
    labels = torch.tensor(labels).long().to(device)

    # l2-norm features
    features /= features.norm(dim=-1, keepdim=True)

    # Set key and values
    model.cache_keys = torch.transpose(features, 1, 0).to(torch.float32).to(device)
    model.cache_values = torch.nn.functional.one_hot(labels, num_classes=classes).to(torch.float32).to(device)

    return model