import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import logging
import time
import json
from datetime import datetime
from torch.optim import Optimizer
from torch import amp
import sacrebleu
from src import generation
import os
import itertools


MODEL_INPUTS = [
    "src_input_ids",
    "tgt_input_ids",
    "src_padding_mask",
    "tgt_padding_mask",
    "tgt_output_ids",
]


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


def generate_metrics(model, batch, max_length, tokenizers, device):

    pred_text = generation.generate_texts(
        model, tokenizers, input_texts=batch["src_text"], max_length=max_length, device=device
    )

    sacrebleu_score = sacrebleu.corpus_bleu(
        pred_text,
        batch["tgt_text"],
        smooth_method="exp",
        smooth_value=0.0,
        lowercase=False,
        tokenize="intl",
    )

    metrics = {
        "bleu": sacrebleu_score.score,
    }

    return metrics


def validation_step(
    model, dataloader, criterion, device, tokenizers, validation_accum_steps, step_no, max_length
):

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):

        # Move batch to device
        batch = {
            k: v.to(device, non_blocking=True) if k in MODEL_INPUTS else v for k, v in batch.items()
        }

        # Forward pass
        with torch.no_grad():
            logits = model(
                batch["src_input_ids"],
                batch["tgt_input_ids"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), batch["tgt_output_ids"].reshape(-1)
            )

        batch_tokens = batch["src_input_ids"].ne(tokenizers["en"].pad_token_id).sum().item()
        metrics = generate_metrics(model, batch, max_length, tokenizers, device)

        if (batch_idx + 1) % validation_accum_steps == 0:
            break

    elapsed_time = time.time() - start_time

    result = {}
    result["type"] = "validation"
    result["step_no"] = step_no
    result["num_tokens"] = batch_tokens
    result["tokens_per_sec"] = batch_tokens / elapsed_time
    result["loss"] = loss.item()
    result["token_length"] = batch["src_input_ids"].shape[1]
    result["bleu"] = metrics["bleu"]

    return result


def train_loop(
    model,
    dataloaders,
    criterion,
    optimiser,
    lr_scheduler,
    num_steps,
    tokenizers,
    grad_accum_steps,
    device,
    checkpoint_steps,
    run_path,
    scaler,
    validation_steps,
    validation_accum_steps,
    max_length,
    results,
    step_no,
):

    # Initialise values - need to do learning rate, optimiser state, scaler state loading here too
    batch_tokens = 0
    minibatch_tokens = 0
    start_time = time.time()
    total_loss = 0.0
    minibatch_idx = step_no * grad_accum_steps
    train_dataloader = itertools.islice(dataloaders["train"], minibatch_idx, None)
    optimiser.zero_grad(set_to_none=True)

    model.train()
    for minibatch_idx, batch in enumerate(train_dataloader, start=minibatch_idx):

        # Move batch to device
        batch = {
            k: v.to(device, non_blocking=True) if k in MODEL_INPUTS else v for k, v in batch.items()
        }

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
        minibatch_tokens = batch["src_input_ids"].ne(tokenizers["en"].pad_token_id).sum().item()
        batch_tokens += minibatch_tokens

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        total_loss += scaled_loss.item()

        # Gradient accumulation
        if (minibatch_idx + 1) % grad_accum_steps != 0:
            continue

        # Step optimiser and scheduler
        scaler.step(optimiser)
        scaler.update()
        lr_scheduler.step()
        optimiser.zero_grad(set_to_none=True)

        # Increment step counter FIRST
        step_no += 1

        # Logging with the completed step number
        elapsed_time = time.time() - start_time
        result = {}
        result["type"] = "train"
        result["step_no"] = step_no
        result["num_tokens"] = batch_tokens
        result["tokens_per_sec"] = batch_tokens / elapsed_time
        result["learning_rate"] = lr_scheduler.get_last_lr()[0]
        result["loss"] = total_loss
        result["token_length"] = batch["src_input_ids"].shape[1]
        results.append(result)
        logging.info(result)

        # Reset counters
        batch_tokens = 0
        start_time = time.time()
        total_loss = 0.0

        # Validation step
        if step_no % validation_steps == 0:
            validation_result = validation_step(
                model,
                dataloaders["test"],
                criterion,
                device,
                tokenizers,
                validation_accum_steps,
                step_no,
                max_length,
            )
            logging.info(validation_result)
            results.append(validation_result)

        # Checkpointing
        if step_no % checkpoint_steps == 0:
            save_checkpoint(model, optimiser, lr_scheduler, scaler, run_path, step_no, results)

        # Stop after num_steps
        if step_no >= num_steps:
            break

        # Delete tensors to free up memory
        del batch
        del logits
        del loss
        del scaled_loss

    return results


def save_checkpoint(model, optimiser, lr_scheduler, scaler, run_path, step_no, results):
    checkpoints_path = f"{run_path}/checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step_no": step_no,
        "results": results,
    }

    torch.save(checkpoint, f"{checkpoints_path}/{step_no}.pt")
    logging.info(f"Checkpoint saved at step {step_no}")
    return None


def load_checkpoint(run_path, step_no, device=None):
    checkpoints_path = f"{run_path}/checkpoints"
    # Load checkpoint with device mapping if device is specified
    if device is not None:
        checkpoint = torch.load(f"{checkpoints_path}/{step_no}.pt", map_location=device)
    else:
        checkpoint = torch.load(f"{checkpoints_path}/{step_no}.pt")

    logging.info(f"Checkpoint loaded from step {step_no}")
    return checkpoint


def create_training_objects(model, train_config):

    # Define loss function, optimiser, and scheduler
    device = torch.device(train_config["train"]["device"])
    criterion = nn.CrossEntropyLoss(
        reduction="mean",
        label_smoothing=train_config["train"]["label_smoothing"],
        ignore_index=train_config["tokenizers"]["pad_token_id"],
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["train"]["learning_rate"],
        betas=train_config["train"]["adam_betas"],
        eps=train_config["train"]["adam_eps"],
        weight_decay=0.01,
    )

    lr_scheduler = WarmupInverseSquareRootLR(optimiser, train_config["train"]["warm_up_steps"])

    scaler = amp.GradScaler()

    return criterion, optimiser, lr_scheduler, scaler


def create_run(model, run_path, config, tokenizers):
    os.makedirs(run_path, exist_ok=True)
    json.dump(config, open(run_path + "/config.json", "w"), indent=2)
    tokenizers["en"].save_pretrained(f"{run_path}/tokenizer_en")
    tokenizers["cy"].save_pretrained(f"{run_path}/tokenizer_cy")
    results = []

    # Move model to device before creating training objects
    device = torch.device(config["train"]["device"])
    model.to(device)

    criterion, optimiser, lr_scheduler, scaler = create_training_objects(model, config)

    return model, criterion, optimiser, lr_scheduler, scaler, results, 0


def load_run(run_path, model, train_config):
    # Check for existing checkpoints
    checkpoints = [f for f in os.listdir(run_path + "/checkpoints")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_path}")

    checkpoint_latest_step = max([int(checkpoint.split(".")[0]) for checkpoint in checkpoints])

    # Move model to device BEFORE loading checkpoint
    device = torch.device(train_config["train"]["device"])
    model.to(device)

    checkpoint = load_checkpoint(run_path, checkpoint_latest_step, device)

    # Create and load training objects
    criterion, optimiser, lr_scheduler, scaler = create_training_objects(model, train_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    results = checkpoint["results"]
    # Use the step_no from the checkpoint to ensure consistency
    step_no = checkpoint["step_no"]

    return model, criterion, optimiser, lr_scheduler, scaler, results, step_no


def get_run(config, model, tokenizers):
    run_dir = config["locations"]["run_dir"]
    resume_run = config["train"].get("resume_run")

    if resume_run and os.path.exists(resume_run):
        run_path = resume_run
        model, criterion, optimiser, lr_scheduler, scaler, results, step_no = load_run(
            run_path, model, config
        )
    else:
        run_name = config.get("name", "run")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = f"{run_dir}/{run_name}_{timestamp}"
        model, criterion, optimiser, lr_scheduler, scaler, results, step_no = create_run(
            model, run_path, config, tokenizers
        )

    return run_path, model, criterion, optimiser, lr_scheduler, scaler, results, step_no


def train(model: nn.Module, dataloaders, tokenizers, config):

    # Set up run
    run_path, model, criterion, optimiser, lr_scheduler, scaler, results, step_no = get_run(
        config, model, tokenizers
    )

    # If step_no >= num_steps, training is complete
    if step_no >= config["train"]["num_steps"]:
        logging.info("Training already complete.")
        return results

    train_config = config["train"]

    # Model is already moved to device in get_run, so we don't need to move it again
    device = torch.device(train_config["device"])

    # Compile model
    if train_config.get("compile_model", False):
        model.compile(fullgraph=True)

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
        tokenizers,
        grad_accum_steps,
        device,
        train_config["checkpoint_steps"],
        run_path,
        scaler,
        validation_steps=train_config["validation_steps"],
        validation_accum_steps=train_config["validation_accum_steps"],
        max_length=config["model"]["max_length"],
        results=results,
        step_no=step_no,
    )
    return results
