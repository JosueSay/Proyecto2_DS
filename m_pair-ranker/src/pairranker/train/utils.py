import os, time, math, random, logging
import numpy as np
import torch

def setSeed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensureDirs(path: str):
    os.makedirs(path, exist_ok=True)

def nowStr():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def getLr(optimizer):
    return optimizer.param_groups[0]["lr"]

def gpuMemMB():
    if torch.cuda.is_available():
        return float(torch.cuda.memory_allocated() / (1024**2))
    return 0.0

def gradGlobalNorm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2)
            tot += float(n.item() ** 2)
    return math.sqrt(tot) if tot > 0 else 0.0

def makeSdpaCtx():
    import contextlib
    return contextlib.nullcontext()

def setupGpuLogging():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("gpu: %s | cuda %s", torch.cuda.get_device_name(0), torch.version.cuda)
