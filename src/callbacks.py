from transformers import TrainerCallback

class EpochLoggerCallback(TrainerCallback):
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Starting epoch {int(state.epoch) + 1}/{state.num_train_epochs} "
              f"(lr={args.learning_rate})")

class TrainLoggerCallback(TrainerCallback):
    
    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training complete! Best metric: {state.best_metric} at step {state.best_global_step}")