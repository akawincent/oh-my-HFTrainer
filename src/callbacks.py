from transformers import TrainerCallback


class EpochLoggerCallback(TrainerCallback):
    def _get_current_lr(self, **kwargs):
        lr_scheduler = kwargs.get("lr_scheduler")
        if lr_scheduler is not None:
            last_lr = lr_scheduler.get_last_lr()
            if last_lr:
                return last_lr[0]

        optimizer = kwargs.get("optimizer")
        if optimizer is not None and optimizer.param_groups:
            return optimizer.param_groups[0]["lr"]

        return None

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_lr = self._get_current_lr(**kwargs)
        lr_text = f"{current_lr:g}" if current_lr is not None else f"{args.learning_rate:g}"
        print(f">>>> Starting epoch {int(state.epoch) + 1}/{state.num_train_epochs} "
              f"[Current learning rate = {lr_text}]")


class TrainLoggerCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training complete! Best metric: {state.best_metric} at step {state.best_global_step}")
