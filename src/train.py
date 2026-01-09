import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import logging
import time
import json
from torch.optim import Optimizer
from torch import amp
from src import utils
import sacrebleu
from src import generation
import os


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


def generate_metrics(model, batch, max_length, tokenizers):

    pred_text = generation.generate_texts(
        model, tokenizers, input_texts=batch["src_text"], max_length=max_length, device="cuda"
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

        elapsed_time = time.time() - start_time

        metrics = generate_metrics(model, batch, max_length, tokenizers)

        if (batch_idx + 1) % validation_accum_steps == 0:
            break

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

    model.train()
    optimiser.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(dataloaders["train"]):

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
        if (batch_idx + 1) % grad_accum_steps != 0:
            continue

        # Logging
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
            save_checkpoint(model, run_path, step_no, results)

        # Stop after num_steps
        if step_no >= num_steps:
            break

        # Delete tensors to free up memory
        del batch
        del logits
        del loss
        del scaled_loss

    return results


def save_checkpoint(model, run_path, step_no, results):
    checkpoints_path = f"{run_path}/checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)
    torch.save(model.state_dict(), f"{checkpoints_path}/{step_no}.pt")
    json.dump(results, open(f"{run_path}/results.json", "w"))
    logging.info(f"Checkpoint saved at step {step_no}")


def load_checkpoint(model, run_path, step_no):
    checkpoints_path = f"{run_path}/checkpoints"
    model.load_state_dict(torch.load(f"{checkpoints_path}/{step_no}.pt"))
    logging.info(f"Checkpoint loaded from step {step_no}")
    return model


def create_run(run_path, config_resolved):
    os.makedirs(run_path, exist_ok=True)
    json.dump(config_resolved, open(run_path + "/config.json", "w"), indent=2)
    results = []
    json.dump(results, open(f"{run_path}/results.json", "w"))
    return results


def load_run(run_path, model):
    results = json.load(open(f"{run_path}/results.json", "r"))
    results_step_no_latest = (
        max([result["step_no"] for result in results if result.get("type") == "train"])
        if results
        else 0
    )
    checkpoints = (
        os.listdir(f"{run_path}/checkpoints") if os.path.isdir(f"{run_path}/checkpoints") else []
    )
    if not checkpoints:
        return model, results

    checkpoint_latest_step = max([int(checkpoint.split(".")[0]) for checkpoint in checkpoints])
    model = load_checkpoint(model, run_path, checkpoint_latest_step)

    # Ensure results correspond to loaded checkpoint
    if results_step_no_latest != checkpoint_latest_step:
        logging.warning(
            "Warning: Loaded checkpoint step does not match latest results step, truncating results."
        )
        results = [r for r in results if r.get("step_no", -1) <= checkpoint_latest_step]

    return model, results


def get_run(config, dataloaders_hash, model):
    config_resolved = {
        **config,
        "dataloaders_hash": dataloaders_hash,
    }
    run_hash = utils.fingerprint(config_resolved)
    run_path = f"{config["locations"]["run_dir"]}/{run_hash}"
    if os.path.exists(run_path):
        model, results = load_run(run_path, model)
    else:
        results = create_run(run_path, config_resolved)
    return run_path, model, results


def train(model: nn.Module, dataloaders, tokenizers, dataloaders_hash, config):

    # Set up run
    run_path, model, results = get_run(config, dataloaders_hash, model)

    train_config = config["train"]

    # Move model to device
    device = torch.device(train_config["device"])
    model.to(device)

    # Compile model
    # model.compile(fullgraph=True)

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

    # Need to reload training state if resuming
    step_no = results[-1].get("step_no", 0) if results else 0

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
