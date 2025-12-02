import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LPpp(torch.nn.Module):
    def __init__(self, zero_shot_prot):
        super().__init__()

        # Textual prototypes
        self.zero_shot_prot = torch.tensor(zero_shot_prot).to(device)
        self.zero_shot_prot /= self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Visual classifier
        self.classifier = torch.nn.Linear(int(zero_shot_prot.shape[1]), int(zero_shot_prot.shape[0]), bias=True)

        # Blending parameter between image and text knowledge
        self.alpha_vec = torch.autograd.Variable(0 * torch.ones(1, int(zero_shot_prot.shape[1])))
        self.alpha_vec.requires_grad = True

    def forward(self, features, act=True):
        
        features = features.to(device)

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = self.zero_shot_prot.clone() / self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Vision logits
        vision_logits = self.classifier(image_features_norm)

        # Textual logits
        text_logits = image_features_norm @ prototypes_norm.t()

        # Combination
        logits = vision_logits + torch.ones(features.shape[0], 1).to(features.dtype).to(device) @ self.alpha_vec * text_logits

        if act:
            return torch.softmax(logits, dim=-1)
        else:
            return logits
