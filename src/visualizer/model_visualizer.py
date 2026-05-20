import torch.nn as nn

def print_model_tree(model: nn.Module, indent: int = 0):
    """
    Pretty print model architecture as a tree (like torch summary)
    """
    space = "  " * indent

    for name, module in model.named_children():
        print(f"{space}({name}): {module.__class__.__name__}")

        # nếu module có children → recurse
        if len(list(module.children())) > 0:
            print_model_tree(module, indent + 1)

