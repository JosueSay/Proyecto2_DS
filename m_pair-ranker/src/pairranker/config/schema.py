from typing import Any

# esquema cerrado del YAML
REQUIRED_SCHEMA: dict[str, Any] = {
    "model": {
        "name": str,
        "pretrained_name": str,
        "dropout": float,
        "grad_checkpointing": bool,
        "compile": bool,
    },
    "lengths": {
        "max_len_prompt": int,
        "max_len_resp": int,
    },
    "train": {
        "epochs": int,
        "batch_size": int,
        "grad_accum": int,
        "lr": float,
        "weight_decay": float,
        "warmup_ratio": float,
        "scheduler": ("cosine", "linear"),
        "clip_norm": float,
        "seed": int,
        "amp": ("bf16", "fp16", "false"),
        "num_workers": int,
    },
    "data": {
        "train_csv": str,
        "valid_csv": str,
        "use_label": bool,
        "use_clean_cols": bool,
        "shuffle": bool,
        "val_batch_size": int,
        "pin_memory": bool,
        "prompt_col": str,
        "respA_col": str,
        "respB_col": str,
    },
    "dataloader": {
        "prefetch_factor_train": int,
        "prefetch_factor_val": int,
        "persistent_workers": bool,
    },
    "loss": {
        "type": ("cross_entropy", "bradley_terry"),
        "label_smoothing": float,
        "class_weights": list,
    },
    "eval": {
        "bt_temp": float,
        "tie_tau": float,
        "tie_alpha": float,
    },
    "logging": {
        "reports_dir": str,
        "runs_dir": str,
        "step_csv": str,
        "epoch_csv": str,
        "alerts_csv": str,
        "confusion_csv": str,
        "class_report_csv": str,
        "preds_sample_csv": str,
        "pred_distributions_csv": str,
        "val_pred_tpl": str,
        "token_budget_tpl": str,
        "run_config_used": str,
        "step_interval": int,
    },
    "monitor": {
        "detect_collapse": bool,
        "save_best_by": str,
        "save_last": bool,
        "verbose": bool,
    },
    "early_stopping": {
        "metric": str,
        "mode": ("max", "min"),
        "patience": int,
    },
    "env": {
        "tokenizers_parallelism": bool,
        "cuda_launch_blocking": int,          # 0 o 1
        "pytorch_cuda_alloc_conf": str,
        "hf_home": str,
        "use_slow_tokenizer": bool,
    },
}
