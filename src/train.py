import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import logging
import time
import json
from torch.optim import Optimizer
from torch import amp


class WarmupInverseSquareRootLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: int,
        last_epoch: int = -1,
    ) -> None:

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warm_up_steps = warm_up_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step_no = self.last_epoch + 1
        if self.last_epoch < self.warm_up_steps:
            lrs = [base_lr * (step_no) / self.warm_up_steps for base_lr in self.base_lrs]
        else:
            lrs = [
                base_lr * (self.warm_up_steps**0.5) / ((step_no) ** 0.5)
                for base_lr in self.base_lrs
            ]
        return lrs


def validation_step(dataloaders):
    return None


def train_loop(
    model,
    dataloaders,
    criterion,
    optimiser,
    lr_scheduler,
    num_steps,
    pad_token_id,
    grad_accum_steps,
    device,
    checkpoint_steps,
    model_dir,
    scaler,
):

    optimiser.zero_grad(set_to_none=True)
    results = []
    step_no = 0
    batch_tokens = 0
    minibatch_tokens = 0
    start_time = time.time()
    total_loss = 0.0
    model.train()
    for batch_idx, batch in enumerate(dataloaders["train"]):

        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with amp.autocast(device_type=device.type):

            # Forward pass
            logits = model(
                batch["src_input_ids"],
                batch["tgt_input_ids"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), batch["tgt_output_ids"].reshape(-1)
            )

        # Compute loss and gradients
        minibatch_tokens = batch["src_input_ids"].ne(pad_token_id).sum().item()
        batch_tokens += minibatch_tokens

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        total_loss += scaled_loss.item()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps != 0:
            continue

        # Logging
        elapsed_time = time.time() - start_time
        result = {}
        result["step_no"] = step_no
        result["num_tokens"] = batch_tokens
        result["tokens_per_sec"] = batch_tokens / elapsed_time
        result["learning_rate"] = lr_scheduler.get_last_lr()[0]
        result["training_loss"] = total_loss
        result["token_length"] = batch["src_input_ids"].shape[1]
        results.append(result)
        logging.info(result)

        # Step optimiser and scheduler
        scaler.step(optimiser)
        scaler.update()
        lr_scheduler.step()
        optimiser.zero_grad(set_to_none=True)

        # Reset counters
        step_no += 1
        batch_tokens = 0
        start_time = time.time()
        total_loss = 0.0

        # Checkpointing
        if step_no % checkpoint_steps == 0:
            torch.save(model.state_dict(), f"{model_dir}/model_step_{step_no}.pt")
            json.dump(results, open(f"{model_dir}/results.json", "w"))
            logging.info(f"Checkpoint saved at step {step_no}")

        # Stop after num_steps
        if step_no >= num_steps:
            break

        # Delete tensors to free up memory
        del batch
        del logits
        del loss
        del scaled_loss

    return results


def train(model, dataloaders, model_dir, config):

    train_config = config["train"]

    # Move model to device
    device = torch.device(train_config["device"])
    model.to(device)

    # Compile model
    model.compile(fullgraph=True)

    # Define loss function, optimiser, and scheduler
    criterion = nn.CrossEntropyLoss(
        reduction="mean",
        label_smoothing=train_config["label_smoothing"],
        ignore_index=config["tokenizers"]["pad_token_id"],
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=train_config["adam_betas"],
        eps=train_config["adam_eps"],
        weight_decay=0.01,
    )

    lr_scheduler = WarmupInverseSquareRootLR(optimiser, train_config["warm_up_steps"])

    scaler = amp.GradScaler()

    grad_accum_steps = (
        train_config["effective_batch_token_size"] // train_config["minibatch_token_size"]
    )

    results = train_loop(
        model,
        dataloaders,
        criterion,
        optimiser,
        lr_scheduler,
        train_config["num_steps"],
        config["tokenizers"]["pad_token_id"],
        grad_accum_steps,
        device,
        train_config["checkpoint_steps"],
        model_dir,
        scaler,
    )
    return results
