import os


def report_to_wandb(report_to):
    if isinstance(report_to, str):
        return report_to in {"all", "wandb"}

    return report_to is not None and ("all" in report_to or "wandb" in report_to)


def configure_wandb_environment(training_args, wandb_args):
    if not report_to_wandb(training_args.report_to):
        return

    if wandb_args.wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_args.wandb_project
    if wandb_args.wandb_mode:
        os.environ["WANDB_MODE"] = wandb_args.wandb_mode
    if wandb_args.wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_args.wandb_watch
    if wandb_args.wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_args.wandb_log_model
