import torch.nn as nn


class FineTuningModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._orig_mod = model

    def forward(self, x):
        feat = self._orig_mod.forward_features(x)
        return self._orig_mod.forward_head(feat), feat


def reset_head(model, args):
    pool = None
    if hasattr(model, "global_pool"):
        pool = getattr(model, "global_pool")
        if not isinstance(pool, str):
            pool = pool.pool_type
    model.reset_classifier(num_classes=args.num_classes, global_pool=pool)


def create_finetuning_model(model: nn.Module, args) -> nn.Module:
    # reset the classifier/head of the model
    reset_head(model, args)

    # freeze model parameters
    total_params = sum(p.numel() for p in model.parameters())
    if args.linear_probing:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    else:
        num_params_freeze = 0
        for param in model.parameters():
            if num_params_freeze / total_params < args.pct_to_freeze:
                param.requires_grad = False
                num_params_freeze += param.numel()
            else:
                break

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, Trainable params: {trainable_params}")

    return FineTuningModel(model)
