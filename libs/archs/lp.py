import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearSoftmax(torch.nn.Module):
    def __init__(self, D, C, norm=True, weights=None, logit_scale=None):
        super().__init__()
        self.D = D       # Features dimension.
        self.C = C       # Number of classes.
        self.norm = norm # Apply l2-norm to the feature space.

        # Init linear weights.
        if weights is not None:
            self.prototypes = torch.tensor(weights)
        else:
            self.prototypes = torch.nn.init.kaiming_normal_(torch.empty([C, D]))

        # Set class prototypes as trainable parameters.
        self.prototypes = torch.nn.Parameter(self.prototypes)

        # Set temperature scaling constant.
        if logit_scale is None:
            logit_scale = 0.
        self.logit_scale = torch.nn.Parameter(torch.tensor(logit_scale))
        self.logit_scale.requires_grad = False

    def forward(self, features, act=True):

        # Get trained prototype
        prototypes = self.prototypes.to(device)
        features = features.to(device)

        # Unit hypersphere normalization
        if self.norm:
            features = features / features.norm(dim=-1, keepdim=True)
            prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

        # Temperature-scaled similarities
        logits = features @ prototypes.t() * self.logit_scale.exp()

        if act:
            return torch.softmax(logits, dim=-1)
        else:
            return logits