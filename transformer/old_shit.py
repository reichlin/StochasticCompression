def transformer_loss(z, z_logits, z_centroids):
    """
    z               ~ B x D x H x W, quantized with z_centroids and masked
    z_tilde_logits  ~ B x L x D x H x W, L is the number of centroids
    z_centroids     ~ L, symbol centroids for z
    """

    z_mask = ~(z.abs() > 0)

    z_target = (z.unsqueeze(0) - z_centroids.view(-1, 1, 1, 1, 1)).abs().argmin(dim=0)  # map z to centroid index
    z_target[z_mask] = -1  # set the masked elements to -1 and ignore that in the loss calculation
    loss = F.cross_entropy(z_logits, z_target, ignore_index=-1)

    return loss


def policy_gradient_loss(k_2, log_pk_2):
    loss = (k_2 * log_pk_2).mean()  # policy gradient with reward = -k

    return loss


def mse_loss(z, z_tilde, z_centroids):

    z_reconstructed = (z_centroids.view(-1, 1, 1, 1, 1) - z_tilde.unsqueeze(0)).abs().argmin(dim=0)
    accuracy = (z == z_reconstructed).sum(dtype=torch.float) / z.shape.numel()
    loss = F.mse_loss(z_tilde, z)

    return loss, accuracy