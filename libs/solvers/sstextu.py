import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sstext_u_solver(features_a, labels_a, model, features_u, classes, num_iters_mm=3, num_iters_ot=10,
                    ratio_label_marginal_correction=4, true_label_marginal=None):

    # num_iters_mm=3:                    repetitions in Majorize-Minimize solver.
    # num_iters_ot=10:                   repetitions in the Sinkhorn Optimal Transport solver.
    # ratio_label_marginal_correction=4: the imputed label-marginal distribution in categories with missing examples in
    #                                    the support set, as a ratio of the least frequent category.
    # true_label_marginal=None:          oracle, in which the label-marginal distribution of the dataset is known.

    with ((torch.no_grad())):

        # %---------------------------------------------------------------
        # Init, constants.
        features_a = torch.tensor(features_a).float().to(device)
        labels_a = torch.tensor(labels_a).long().to(device)
        features_u = torch.tensor(features_u).float().to(device)
        textual_prototypes = model.prototypes.clone().to(device) # Extract the zero-shot weights from init head.

        # %---------------------------------------------------------------
        # A. Visual embeddings for annotates samples

        # Get labels as one-hot encoding
        ohe = torch.nn.functional.one_hot(labels_a, num_classes=classes).float()

        # Compute visual embedding
        vision_mu_a = torch.einsum('ij,ik->jk', ohe, features_a)

        # Adjust relative importance of text prototypes
        lambda_text = (1 / (torch.sum(ohe, dim=0, keepdim=True)).pow(1)).t()

        # Set temperature scaling parameter, tau.
        tau = (1 / model.logit_scale.exp())

        # %---------------------------------------------------------------
        # B. Expectation-Maximization on unlabeled data.

        # Estimate label-marginal distribution
        if true_label_marginal is None:

            # Empirical label dist. from the few shots.
            label_dist = (ohe.sum(0) / ohe.sum())

            # Correction - in case there are zeros for some absent categories.
            idx = torch.where(label_dist == 0)[0].cpu().numpy()
            if len(idx) > 0:
                min_pos_label_frequency = min(label_dist[label_dist > 0])
                label_dist += (label_dist == 0) * (min_pos_label_frequency / ratio_label_marginal_correction)
                label_dist = label_dist / label_dist.sum()
        else:
            label_dist = torch.tensor(true_label_marginal)

        # Init weights
        new_mu = textual_prototypes
        model.prototypes.data = new_mu.to(device)

        # In case of not using BMM iterations, output the SS-Text+ solver solution
        if num_iters_mm==0:
            new_mu = vision_mu_a * (1 / features_a.shape[0]) * (1 / tau) * (1 / (2*lambda_text)) + \
                     textual_prototypes
            model.prototypes.data = new_mu.to(device)
            return model
        else:
            new_mu = textual_prototypes
            model.prototypes.data = new_mu.to(device)

        # Run BMM optimizer.
        for i in range(num_iters_mm):

            # 1.Expectation: compute pseudo-labels on unlabeled data trough optimal transport.

            # Get logits.
            scores_u = model(features_u, act=False)

            # Compute optimal transport to get class assignments.
            codes = distributed_sinkhorn(scores_u, num_iters=num_iters_ot, r=label_dist)

            # 2. Maximization: adjust weights.

            # Compute unlabeled prototypes
            vision_mu_u = torch.einsum('ij,ik->jk', codes, features_u)

            # Aggregate all weights
            new_mu = vision_mu_a * (1 / features_a.shape[0]) * (1 / tau) * (1 / (2*lambda_text)) + \
                     vision_mu_u * (1 / features_u.shape[0]) * (1 / tau)                         + \
                     textual_prototypes
            model.prototypes.data = new_mu.to(device)

    return model


def distributed_sinkhorn(similarities, epsilon=1.0, num_iters=3, r=None, c=None):

    if num_iters==0:
        return torch.softmax(similarities / epsilon, dim=-1)

    similarities = similarities.to(device)
    Q = torch.exp(similarities / epsilon).t()  # Q is K-by-B for consistency with notations
    K, B = Q.shape  # Number of clusters and batch size

    if r is None:
        r = torch.ones(K).to(similarities.device) / K
    if c is None:
        c = torch.ones(B).to(similarities.device) / B

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(num_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q *= r.unsqueeze(1)

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q *= c.unsqueeze(0)

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    Q = Q.t()

    return Q