import json
import math
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path

import optuna
import transformers

from arguments import DataArguments, HPOArguments, ModelArguments, TrainingArguments, WandbArguments
from wandb_utils import configure_wandb_environment


def save_best_hyperparameters(output_dir: str | None, best_trial: optuna.trial.FrozenTrial):
    output_path = Path(output_dir or "./outputs") / "best_hyperparameters.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": str(best_trial.number),
        "objective": best_trial.value,
        "hyperparameters": best_trial.params,
        "trial_output_dir": best_trial.user_attrs.get("trial_output_dir"),
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved best hyperparameters to {output_path}")


def bool_to_arg(value: bool) -> str:
    return "True" if value else "False"


def serialize_arg_value(value) -> str:
    if isinstance(value, bool):
        return bool_to_arg(value)
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def add_arg(args: list[str], flag: str, value):
    if value is None:
        return

    if isinstance(value, (list, tuple)):
        if not value:
            return
        args.append(flag)
        args.extend(serialize_arg_value(item) for item in value)
        return

    args.extend([flag, serialize_arg_value(value)])


def load_trial_metric(output_dir: Path, metric_name: str) -> float:
    metric_files = [
        output_dir / "all_results.json",
        output_dir / "eval_results.json",
    ]

    for metric_file in metric_files:
        if not metric_file.exists():
            continue

        with metric_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if metric_name in payload:
            metric_value = float(payload[metric_name])
            if math.isnan(metric_value) or math.isinf(metric_value):
                raise RuntimeError(f"Metric {metric_name} in {metric_file} is not finite: {metric_value}")
            return metric_value

    raise RuntimeError(
        f"Unable to find metric '{metric_name}' in trial output directory {output_dir}. "
        f"Checked files: {[str(path) for path in metric_files]}"
    )


def build_trial_command(
    repo_root: Path,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    hpo_args: HPOArguments,
    wandb_args: WandbArguments,
    trial: optuna.Trial,
    trial_output_dir: Path,
) -> list[str]:
    sampled_params = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            hpo_args.hpo_learning_rate_min,
            hpo_args.hpo_learning_rate_max,
            log=True,
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size",
            hpo_args.hpo_batch_size_choices,
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps",
            hpo_args.hpo_gradient_accumulation_choices,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            hpo_args.hpo_weight_decay_min,
            hpo_args.hpo_weight_decay_max,
        ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio",
            hpo_args.hpo_warmup_ratio_min,
            hpo_args.hpo_warmup_ratio_max,
        ),
    }

    launch_config = (repo_root / hpo_args.hpo_accelerate_config).resolve()
    train_entry = (repo_root / hpo_args.hpo_train_entry_file).resolve()
    trial_run_name = f"{training_args.run_name or 'optuna-hpo'}-trial-{trial.number:03d}"

    args = [
        "accelerate",
        "launch",
        "--config_file",
        str(launch_config),
        str(train_entry),
    ]

    add_arg(args, "--model_name_or_path", model_args.model_name_or_path)
    add_arg(args, "--model_dtype", model_args.model_dtype)
    add_arg(args, "--model_max_length", model_args.model_max_length)
    add_arg(args, "--attn_implementation", model_args.attn_implementation)

    add_arg(args, "--dataset_name", data_args.dataset_name)
    add_arg(args, "--test_split", data_args.test_split)

    add_arg(args, "--output_dir", trial_output_dir)
    add_arg(args, "--optim", training_args.optim)
    add_arg(args, "--bf16", training_args.bf16)
    add_arg(args, "--num_train_epochs", training_args.num_train_epochs)
    add_arg(args, "--gradient_checkpointing", training_args.gradient_checkpointing)
    add_arg(args, "--dataloader_num_workers", training_args.dataloader_num_workers)
    add_arg(args, "--dataloader_pin_memory", training_args.dataloader_pin_memory)
    add_arg(args, "--logging_steps", training_args.logging_steps)
    add_arg(args, "--eval_strategy", training_args.eval_strategy)
    add_arg(args, "--save_strategy", training_args.save_strategy)
    add_arg(args, "--load_best_model_at_end", training_args.load_best_model_at_end)
    add_arg(args, "--metric_for_best_model", training_args.metric_for_best_model)
    add_arg(args, "--greater_is_better", training_args.greater_is_better)
    add_arg(args, "--report_to", training_args.report_to)
    add_arg(args, "--run_name", trial_run_name)
    add_arg(args, "--max_grad_norm", training_args.max_grad_norm)

    if training_args.deepspeed:
        add_arg(args, "--deepspeed", training_args.deepspeed)

    add_arg(args, "--wandb_project", wandb_args.wandb_project)
    add_arg(args, "--wandb_mode", wandb_args.wandb_mode)
    add_arg(args, "--wandb_watch", wandb_args.wandb_watch)
    add_arg(args, "--wandb_log_model", wandb_args.wandb_log_model)

    add_arg(args, "--learning_rate", sampled_params["learning_rate"])
    add_arg(args, "--per_device_train_batch_size", sampled_params["per_device_train_batch_size"])
    add_arg(args, "--gradient_accumulation_steps", sampled_params["gradient_accumulation_steps"])
    add_arg(args, "--weight_decay", sampled_params["weight_decay"])
    add_arg(args, "--warmup_ratio", sampled_params["warmup_ratio"])

    return args


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            HPOArguments,
            WandbArguments,
        )
    )
    model_args, data_args, training_args, hpo_args, wandb_args = parser.parse_args_into_dataclasses()

    configure_wandb_environment(training_args, wandb_args)

    repo_root = Path(__file__).resolve().parent.parent
    base_output_dir = Path(training_args.output_dir or "./outputs").resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction=hpo_args.hpo_direction)

    def objective(trial: optuna.Trial) -> float:
        trial_output_dir = base_output_dir / f"trial-{trial.number:03d}"
        if trial_output_dir.exists():
            shutil.rmtree(trial_output_dir)
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        command = build_trial_command(
            repo_root=repo_root,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            hpo_args=hpo_args,
            wandb_args=wandb_args,
            trial=trial,
            trial_output_dir=trial_output_dir,
        )

        env = os.environ.copy()
        print(f"[HPO] Starting trial {trial.number} with output dir {trial_output_dir}")
        print(f"[HPO] Command: {' '.join(command)}")

        completed = subprocess.run(command, cwd=repo_root, env=env)
        if completed.returncode != 0:
            raise RuntimeError(f"Trial {trial.number} failed with exit code {completed.returncode}")

        metric_value = load_trial_metric(trial_output_dir, hpo_args.hpo_metric)
        trial.set_user_attr("trial_output_dir", str(trial_output_dir))
        print(f"[HPO] Trial {trial.number} completed with {hpo_args.hpo_metric}={metric_value}")
        return metric_value

    study.optimize(objective, n_trials=hpo_args.hpo_n_trials, catch=(RuntimeError,))

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if not completed_trials:
        raise RuntimeError("No HPO trials completed successfully.")

    save_best_hyperparameters(training_args.output_dir, study.best_trial)
