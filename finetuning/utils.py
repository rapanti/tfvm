import torch.nn as nn


class FineTuningModel(nn.Module):
    """
    A wrapper of the original model for fine-tuning.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model (nn.Module): The original model.
        """
        super().__init__()
        self._orig_mod = model

    def forward(self, x):
        """
        Forward a batch of input data through the model.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            logits (torch.Tensor): The output of the model.
            pre_logits (torch.Tensor): The output of the model before the final
                classification layer.
        """
        # Compute the feature representation of the input data
        feat = self._orig_mod.forward_features(x)

        # Compute the output of the model before the final classification layer
        pre_logits = self._orig_mod.forward_head(feat, pre_logits=True)

        # Compute the output of the model
        logits = self._orig_mod.forward_head(feat)

        return logits, pre_logits


def reset_head(model, args):
    """
    Reset the head of a model to the given number of classes.

    Args:
        model (nn.Module): The model to reset the head of.
        args: The arguments containing the number of classes to reset the head to.

    Returns:
        The model with the head reset to the given number of classes.
    """
    pool = None
    if hasattr(model, "global_pool"):
        pool = getattr(model, "global_pool")
        if not isinstance(pool, str):
            pool = pool.pool_type
    model.reset_classifier(num_classes=args.num_classes, global_pool=pool)


def create_finetuning_model(
    model: nn.Module, args
) -> nn.Module:
    """
    Create a fine-tuning model from a given model.

    Args:
        model (nn.Module): The model to create a fine-tuning model from.
        args: Arguments-object for creating the fine-tuning model.

    Returns:
        nn.Module: The created fine-tuning model.
    """
    # reset the classifier/head of the model
    reset_head(model, args)
    
    total_params: int = sum(param.numel() for param in model.parameters())

    #freeze model parameters
    if args.linear_probing:
        for param in model.parameters():
            param.requires_grad = False
    else:
        num_frozen: int = 0
        for param in model.parameters():
            if num_frozen < args.pct_to_freeze * total_params:
                param.requires_grad = False
                num_frozen += param.numel()
            else:
                break
    # unfreezing classifier/head, always used for finetuning 
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, Trainable params: {trainable_params}")

    return FineTuningModel(model)
