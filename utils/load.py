import os
import torch
import torch.nn as nn
import torch.optim as optim

def continue_training(checkpoint_path, model: nn.Module, optimizer: optim.Optimizer, steps_per_epoch: int = None, target_step: int = None) -> int:
    """
    Resume training from the latest checkpoint (supports both epoch-based and step-based saving).

    Args:
        checkpoint_path: path to folder containing checkpoint files
        model: model instance (can be nn.Module or DDP)
        optimizer: optimizer instance
        steps_per_epoch: optional, used to convert step number -> epoch number
        target_step: optional, specify a particular checkpoint to resume from

    Returns:
        int: current_epoch (for resuming loop)
    """

    model_files = {}
    optim_files = {}

    # === 1️⃣ Duyệt qua tất cả checkpoint ===
    for f in os.listdir(checkpoint_path):
        if not f.endswith(".pt"):
            continue

        if "interrupt" in f or "best" in f:
            print(f"[WARN] Skipping invalid checkpoint: {f}")
            continue

        # Hỗ trợ cả dạng checkpoint_step_123.pt hoặc checkpoint_123.pt
        num_str = None
        if "_step_" in f:
            num_str = f.split("_step_")[-1].split(".")[0]
        elif "_" in f:
            num_str = f.split("_")[-1].split(".")[0]

        if num_str is None or not num_str.isdigit():
            print(f"[WARN] Skipping unrecognized file: {f}")
            continue

        step = int(num_str)
        if f.startswith("checkpoint"):
            model_files[step] = f
        elif f.startswith("optimizer"):
            optim_files[step] = f

    # === 2️⃣ Kiểm tra xem có cặp checkpoint hợp lệ không ===
    common_steps = set(model_files.keys()) & set(optim_files.keys())
    if not common_steps:
        print("[INFO] No valid checkpoint found — starting from scratch.")
        return 0

    # === 3️⃣ Chọn checkpoint cần load ===
    if target_step is not None:
        if target_step not in common_steps:
            raise ValueError(f"No matching checkpoint for step {target_step}. Available: {sorted(common_steps)}")
        load_step = target_step
    else:
        load_step = max(common_steps)

    model_file = os.path.join(checkpoint_path, model_files[load_step])
    optim_file = os.path.join(checkpoint_path, optim_files[load_step])

    # === 4️⃣ Load model và optimizer ===
    print(f"✅  Resuming training from checkpoint: step {load_step}")
    state_dict = torch.load(model_file, map_location="cpu")

    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    optimizer.load_state_dict(torch.load(optim_file, map_location="cpu"))

    # === 5️⃣ Quy đổi step -> epoch nếu có steps_per_epoch ===
    if steps_per_epoch:
        resumed_epoch = load_step // steps_per_epoch + 1
    else:
        # fallback nếu không có, tránh việc model dừng luôn
        resumed_epoch = 0

    return resumed_epoch
