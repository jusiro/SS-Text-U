import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TIPAd(torch.nn.Module):
    def __init__(self, zero_shot_prot):
        super().__init__()

        # Textual prototypes
        self.zero_shot_prot = torch.tensor(zero_shot_prot).to(device)
        self.zero_shot_prot /= self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Visual classifier
        self.alpha = 1.0
        self.beta = 1.0

    def forward(self, features, act=True):

        features = features.to(device)

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = self.zero_shot_prot.clone() / self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Vision logits
        affinity = image_features_norm @ self.cache_keys
        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values

        # Textual logits
        text_logits = image_features_norm @ prototypes_norm.t()

        # Combination
        logits = text_logits + cache_logits * self.alpha

        if act:
            return torch.softmax(logits, dim=-1)
        else:
            return logits