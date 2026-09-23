import torch


def realign_parameters(module: torch.nn.Module) -> None:
    """Copy memory-mapped weights into freshly allocated (aligned) tensors.

    Some safetensors checkpoints (ms-marco-MiniLM-L-6-v2 among them) store tensors
    at offsets that aren't float-aligned. When transformers memory-maps them, CPU
    BLAS kernels silently return NaN for matmuls against those weights.
    """
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            if tensor.untyped_storage().data_ptr() % tensor.element_size():
                tensor.data = tensor.data.clone()
