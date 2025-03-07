#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import logging
import torch
import datasets
import itertools
import transformers
from typing import Callable
from collections import defaultdict, OrderedDict
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import reduce, tqdm, DataLoaderConfiguration
from transformers import PreTrainedModel, pipeline, AutoTokenizer, default_data_collator
from transformers.generation.utils import ModelOutput
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F

from llmart import config, data, optim, transforms, losses, schedulers
from llmart import TaggedTokenizer, AdversarialAttack, AttackPrompt


def run_attack(cfg: config.LLMartConf) -> dict:
    """Find an attack on a given language model and dataset.

    Perform input optimization using the specified configuration to attack
    a language model and generate an adversarial example.

    Args:
        cfg: Configuration object containing model, attack, and data parameters.

    Returns:
        results: Dictionary containing various results and metrics.
    """

    # Seed
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.use_deterministic_algorithms, warn_only=True)

    # Setup Tensorboard
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=cfg.output_dir,
        dataloader_config=DataLoaderConfiguration(
            split_batches=cfg.data.split_batches
            if cfg.data.split_batches is not None
            else (True if cfg.data.n_train > 1 else False)
        ),
        step_scheduler_with_optimizer=False,
    )
    accelerator.init_trackers(cfg.experiment_name, config=cfg.asdict(flatten=True))

    # Setup logging
    transformers.logging.set_verbosity_error()
    datasets.utils.disable_progress_bars()
    log = get_logger(__name__)
    log.info(f"{cfg.output_dir=}")

    # Create attack and responses dataset transforms
    attack_prompt = transforms.from_config(cfg.attack)
    mask_completion = transforms.from_config(cfg.response)
    assert isinstance(attack_prompt, AttackPrompt)

    # Create adversarial tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.chat_template = cfg.model.chat_template or tokenizer.chat_template
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer = TaggedTokenizer(
        tokenizer,  # type: ignore
        tags=attack_prompt.tags + mask_completion.tags,
    )

    # Create data, apply attack transforms to it
    with accelerator.main_process_first():
        ds = data.from_config(
            cfg.data,
            tokenizer=tokenizer,
            mark_prompt=attack_prompt,
            mark_completion=mask_completion,
        )

    for name in filter(lambda name: len(ds[name]), ds):
        log.info(f"{name} data:")
        for i, (input_ids, input_map, attention_mask) in enumerate(
            zip(
                ds[name]["input_ids"], ds[name]["input_map"], ds[name]["attention_mask"]
            )
        ):
            input_ids = list(itertools.compress(input_ids, attention_mask))
            input_map = list(itertools.compress(input_map, attention_mask))
            log.info(f"{i:4d}: {tokenizer.pretty_decode(input_ids, input_map)}")

    # Load demo models
    pipe = pipeline(
        task=cfg.model.task,
        model=cfg.model.name,
        revision=cfg.model.revision,
        device=cfg.model.device,
        device_map=cfg.model.device_map,
        trust_remote_code=True,
        torch_dtype=cfg.model.torch_dtype,
        tokenizer=tokenizer,
    )
    model = pipe.model
    model.requires_grad_(False)

    # Optimize attack
    step, attack = 0, None
    results = dict()
    if len(attack_prompt.elements) > 0:
        step, attack, train_results = train(
            ds, attack_prompt, tokenizer, model, cfg, accelerator, log
        )
        results.update(train_results)

    # Evaluate test data
    test_dl = DataLoader(ds["test"], collate_fn=default_data_collator)  # type: ignore
    if len(test_dl):
        log.info(f"== TEST @ {step} ==")
        outputs = evaluate(test_dl, tokenizer, model, attack, log, cfg.max_new_tokens)
        outputs = {f"eval/test_{key}": value for key, value in outputs.items()}
        results.update(outputs)
        accelerator.log(outputs, step=step)

    accelerator.end_training()

    return results


def train(
    ds: datasets.DatasetDict,
    attack_prompt: AttackPrompt,
    tokenizer: TaggedTokenizer,
    model: PreTrainedModel,
    cfg: config.LLMartConf,
    accelerator: Accelerator,
    log: logging.Logger | logging.LoggerAdapter,
) -> tuple[int, AdversarialAttack, dict]:
    # Create adversarial attack and losses from tokenized prompt attack
    attack_inits = tokenizer(
        attack_prompt.elements,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    attack = AdversarialAttack(
        attack_inits,
        model.get_input_embeddings(),
        dim=cfg.attack.dim,
        init=tokenizer.good_token_ids if cfg.attack.init == "rand" else cfg.attack.init,
    )
    log.info(f"{attack=}")

    # Return attack if no training planned
    if cfg.steps <= 0:
        return 0, attack, dict()

    # Dataloaders
    eval_dl = DataLoader(ds["train"], collate_fn=default_data_collator)  # type: ignore
    train_dl = DataLoader(
        ds["train"],  # type: ignore
        collate_fn=default_data_collator,
        batch_size=cfg.bs,
        sampler=RandomSampler(
            ds["train"],
            # No replacement for full-batch GD
            replacement=cfg.with_replacement if cfg.bs < len(ds["train"]) else False,
            num_samples=cfg.bs * cfg.steps * accelerator.num_processes,
        ),
    )
    minitrain_dl = DataLoader(ds["minitrain"], collate_fn=default_data_collator)  # type: ignore
    val_dl = DataLoader(ds["val"], collate_fn=default_data_collator)  # type: ignore

    # Create optimizer from adversarial attack parameters
    loss_fn = losses.from_config(cfg.loss)
    optimizer = optim.from_config(
        cfg.optim,
        attack.parameters(),
        ignored_values=tokenizer.bad_token_ids,
        embedding=attack.embedding if cfg.attack.dim == 1 else None,
    )
    scheduler = schedulers.from_config(
        cfg.scheduler,
        optimizer,
    )

    train_dl, model, optimizer, scheduler, attack = accelerator.prepare(
        train_dl, model, optimizer, scheduler, attack
    )

    # Make closure to pass to optimizer
    closure, closure_inputs = make_closure(
        attack,
        model,
        losses.from_config(cfg.closure_loss or cfg.loss),
        is_valid_input=tokenizer.reencodes,
        num_samples=cfg.bs,
        batch_size=cfg.per_device_bs,
        use_kv_cache=cfg.use_kv_cache,
    )

    # For each optimization step
    step, results = 0, dict()
    best_loss = float('inf')
    best_attack_data = None

    for step, inputs in (
        pbar := tqdm(iterable=enumerate(train_dl), total=len(train_dl), desc="steps")
    ):
        optimizer.zero_grad()

        attack_param = attack.module.param if hasattr(attack, 'module') else attack.param
        attack_data = attack_param.data
        model_loss, loss, attack_success, attack_count = 0.0, 0.0, 0, 0
        for micro_inputs in data.microbatch(inputs, micro_batch_size=cfg.per_device_bs):
            # Get adversarial version of inputs and compute loss using differentiable embedding
            micro_inputs = attack(micro_inputs)
            if not tokenizer.reencodes(micro_inputs["input_ids"]).all():
                log.warning("Adversarial inputs do not reencode.")

            outputs = model(
                inputs_embeds=micro_inputs["inputs_embeds"],
                labels=micro_inputs["labels"],
                attention_mask=micro_inputs["attention_mask"],
            )
            local_loss = loss_fn(outputs, micro_inputs["labels"])
            accelerator.backward(local_loss)

            # Accumulate across micro-batches
            model_loss += outputs.loss.detach() * len(micro_inputs["labels"])
            loss += local_loss.detach() * len(micro_inputs["labels"])

            # Keep track of per-token attack success rate
            shift_preds = outputs["logits"].detach()[..., :-1, :].argmax(-1)
            shift_labels = micro_inputs["labels"][..., 1:]
            is_valid = shift_labels != -100
            attack_success += (shift_preds == shift_labels)[is_valid].sum()
            attack_count += is_valid.sum()

        with torch.inference_mode():
            # Accumulate across devices
            if accelerator.split_batches:
                assert isinstance(tokenizer.pad_token_id, int)
                inputs = data.gather_batch_across_processes(
                    inputs,
                    dim=1,
                    pad_first=False,
                    # Use defaultdict to dynamically choose different pad_index values for different input keys
                    pad_index=defaultdict(
                        lambda: 0, input_ids=tokenizer.pad_token_id, labels=-100
                    ),
                )
                loss = reduce(loss, reduction="sum") / len(inputs["labels"])  # type: ignore
                model_loss = reduce(model_loss, reduction="sum") / len(inputs["labels"])  # type: ignore
                attack_success = reduce(attack_success, reduction="sum")  # type: ignore
                attack_count = reduce(attack_count, reduction="sum")  # type: ignore
            success_rate = attack_success / attack_count  # type: ignore

            # Log and update progress bar
            scheduler_var_name = getattr(scheduler.scheduler, "var_name", "lr")
            attack_log = {
                "attack/loss": loss,
                "attack/model_loss": model_loss,
                "attack/success_rate": success_rate,
                f"attack/{scheduler_var_name}": scheduler.get_last_lr()[0],
            }
            results.update(attack_log)
            accelerator.log(attack_log, step=step)
            postfix = OrderedDict(
                {
                    "loss": f"{loss:0.4f}",
                    "success_rate": f"{success_rate:0.3f}",
                    scheduler_var_name: scheduler.get_last_lr()[0],
                }
            )
            if len(optimizer.state.values()) == 1 and (
                swap_count := list(optimizer.state.values())[0].get("swap_count", None)
            ):
                accelerator.log({"attack/swap_count": swap_count}, step=step)
                postfix["swap_count"] = f"{swap_count:d}"
            postfix["mem"] = f"{torch.cuda.max_memory_allocated()/(1024**2):0.3f}MiB"
            pbar.set_postfix(postfix)

            # Save the best attack token
            if loss < best_loss:
                best_loss = loss
                best_attack_data = attack_data

            # Exit attack loop if we found a successful attack across all training examples
            if (
                cfg.early_stop
                and len(eval_dl) == 1
                and torch.allclose(success_rate, torch.tensor(1.0))
            ):
                # NOTE: We use evaluate because model() can differ from model.generate()
                outputs = evaluate(eval_dl, tokenizer, model, attack, max_new_tokens=0)
                if torch.allclose(outputs["attack_success_rate"], torch.tensor(1.0)):
                    break

            # Gather data for step and take step
            closure_inputs.update(inputs)
            optimizer.step(closure)
            scheduler.step(loss)
            step = step + 1

            # Evaluate on minitrain/val/test datasets and save attack
            if len(minitrain_dl) and cfg.val_every and step % cfg.val_every == 0:
                log.info(f"== MINITRAIN @ {step} ==")
                outputs = evaluate(minitrain_dl, tokenizer, model, attack, log)
                outputs = {f"train/{key}": value for key, value in outputs.items()}
                accelerator.log(outputs, step=step)
            if len(val_dl) and cfg.val_every and step % cfg.val_every == 0:
                log.info(f"== VAL @ {step} ==")
                outputs = evaluate(val_dl, tokenizer, model, attack, log)
                outputs = {f"val/{key}": value for key, value in outputs.items()}
                accelerator.log(outputs, step=step)
            if (
                accelerator.is_main_process
                and cfg.save_every
                and step % cfg.save_every == 0
            ):
                attack_path = f"{cfg.output_dir}/attack_{step}.pt"
                torch.save(accelerator.unwrap_model(attack).state_dict(), attack_path)
                log.info(f"{attack_path=}")

    if accelerator.is_main_process:
        attack_path = f"{cfg.output_dir}/attack_{step}.pt"
        torch.save(accelerator.unwrap_model(attack).state_dict(), attack_path)
        log.info(f"{attack_path=}")

    if cfg.keep_best_attack:
        # Update attack.param with the best attack token
        if hasattr(attack, 'module'):
            attack.module.param.data = best_attack_data
        else:
            attack.param.data = best_attack_data

    return step, attack, results


def make_closure(
    attack: torch.nn.Module,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    is_valid_input: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = 1,
    batch_size: int = 1,
    use_kv_cache: bool = False,
):
    """Make a closure/generator suitable for passing to the GCG optimizer.

    Args:
        attack: An AdversarialAttack to apply to closure inputs
        model: A PreTrainedModel to turn attacked inputs into logits
        loss_fn: A nn.Module that turns logits and labels into a loss
        is_valid_input: A Callable returns whether an input is valid
        batch_size: How many samples batch together
        num_samples: Number of sampples in closure inputs
        use_kv_cache: Whether to use the kv_cache when batched=True

    Returns:
        A closure/generator and closure inputs to update before passing closure.
    """
    inputs = {}

    def generator():
        """A generator that accumulates attacks on **a single training example** until
        a desired batch size, and then computes per-attack losses. No loss is computed
        for non-valid attacks.

        Yields:
            List of tuples containing attack indices and their losses.
        """

        param_losses = []
        batch = defaultdict(list)
        kv_cache = None
        kv_cache_len = 0

        while True:
            # Get next attack
            param_idx = yield param_losses

            # If we have a whole batch, or a partial batch and we're stopping,
            # then compute per-example losses
            if (len(batch["param_idx"]) == batch_size) or (
                param_idx is None and len(batch["param_idx"])
            ):
                batch_input_ids = torch.cat(batch["input_ids"])
                batch_attention_mask = torch.cat(batch["attention_mask"])
                batch_kv_cache = kv_cache
                if batch_kv_cache is not None:
                    batch_kv_cache = [
                        tuple(t.expand(len(batch_input_ids), -1, -1, -1) for t in kv)
                        for kv in batch_kv_cache
                    ]
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    past_key_values=batch_kv_cache,
                )

                # Compute per-example loss
                losses = itertools.starmap(
                    loss_fn, zip(outputs["logits"], torch.cat(batch["labels"]))
                )
                param_losses = list(zip(batch["param_idx"], losses))

                # Next iteration will start reaccumulating a batch
                batch = defaultdict(list)
            else:
                param_losses = []

            # If we're stopping, then yield any remaing losses
            if param_idx is None:
                yield param_losses
                break

            # Otherwise, attack inputs and make sure they reencode..
            adv_inputs = attack(inputs)
            if not is_valid_input(adv_inputs["input_ids"]).all():
                continue

            # ...compute past key values...
            if use_kv_cache and kv_cache is None:
                # NOTE: This assumes a batch size of 1
                kv_cache_len = adv_inputs["input_map"].nonzero()[0, 1]
                outputs = model(
                    input_ids=adv_inputs["input_ids"][:, :kv_cache_len],
                    attention_mask=adv_inputs["attention_mask"][:, :kv_cache_len],
                    use_cache=True,
                )
                kv_cache = outputs["past_key_values"]

            # ...and if they do accumulate a batch
            batch["param_idx"].append(param_idx)
            batch["input_ids"].append(adv_inputs["input_ids"][:, kv_cache_len:])
            batch["attention_mask"].append(adv_inputs["attention_mask"])
            batch["labels"].append(adv_inputs["labels"][:, kv_cache_len:])

    def closure():
        """A function that computes the average loss of an attack applied to
        **many training examples**. If any sample is not valid, then an
        infinite loss is returned.

        Returns:
            float: Average loss across the entire attacked training batch
        """

        loss = 0.0
        for micro_inputs in data.microbatch(inputs, micro_batch_size=batch_size):
            adv_inputs = attack(micro_inputs)
            if not is_valid_input(adv_inputs["input_ids"]).all():
                loss = torch.tensor(torch.inf, device=model.device)
                break
            else:
                outputs = model(
                    input_ids=adv_inputs["input_ids"],
                    attention_mask=adv_inputs["attention_mask"],
                    labels=micro_inputs["labels"],
                    use_cache=False,
                )
                micro_loss = loss_fn(outputs, micro_inputs["labels"])
                # Accumulate averages across micro-batches
                # NOTE: Assumes equal distribution of micro-batch across devices
                loss = loss + micro_loss * len(micro_inputs["labels"])

        # Average across the entire training batch
        loss = loss / len(inputs["labels"])

        return loss

    return (generator if num_samples == 1 else closure), inputs


@torch.random.fork_rng(devices=range(torch.cuda.device_count()))
@torch.no_grad()
def evaluate(
    dataloader: DataLoader,
    tokenizer: TaggedTokenizer,
    model: PreTrainedModel,
    attack: AdversarialAttack | None,
    log: logging.Logger | logging.LoggerAdapter | None = None,
    max_new_tokens: int = 50,
) -> ModelOutput:
    """Evaluate attack on a dataset against a language model.

    Generates greedily-decoded continuations for the attack applied to each prompt
    in the dataloader, and computes per-attack loss and success rate.

    Args:
        dataloader: DataLoader containing prompts
        tokenizer: Tokenizer for decoding prompts
        model: Language model to generate continuations
        attack: Attack to apply to prompts
        log: Optional logger for outputting results
        max_new_tokens: Maximum number of new tokens to generate (default: 50)

    Returns:
        ModelOutput containing evaluation metrics including:
        - loss: Average loss across evaluation examples
        - attack_success_rate: Proportion of successful token forcings
        - Per-token probabilities and token forcing rankings
        - Attacked prompts and continuations
    """

    outputs = ModelOutput()
    outputs["loss"] = []
    outputs["attack_success_rate"] = []

    for i, inputs in enumerate(dataloader):
        assert len(inputs["input_ids"]) == 1
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs = attack(inputs) if attack else inputs
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        response_mask = inputs["response_mask"][0]

        # Decode prompt
        prompt_end = response_mask.nonzero()[0, 0]
        prompt_ids = input_ids[:prompt_end]
        prompt = tokenizer.decode(prompt_ids)
        log.info(f"{prompt=}") if log else None

        # Deterministically generate a response using prompt_ids
        output = model.generate(
            inputs=prompt_ids[None],
            attention_mask=attention_mask[:prompt_end][None],
            do_sample=False,
            temperature=None,
            top_p=None,
            min_new_tokens=len(response_mask) - len(prompt_ids),
            max_new_tokens=max(max_new_tokens, len(response_mask) - len(prompt_ids)),
            return_dict_in_generate=True,
            output_logits=True,
            return_legacy_cache=False,
        )

        # Decode continuation of prompt
        continuation_ids = output.sequences[0]  # type: ignore
        continuation = tokenizer.decode(continuation_ids)

        # Compute loss
        targets = input_ids[prompt_end:]
        continuation_mask = response_mask[prompt_end:]

        logits = torch.cat(output.logits)  # type: ignore
        logits = logits[: len(targets)]

        logits = logits[continuation_mask]
        targets = targets[continuation_mask]
        loss = F.cross_entropy(logits, targets)
        attack_success = (logits.argmax(-1) == targets).sum()
        attack_count = (targets != -100).sum()
        attack_success_rate = attack_success / attack_count

        log.info(
            f"{continuation=} {loss=:0.4f} {attack_success_rate=:0.3f}"
        ) if log else None

        # Log prob and rank of targets
        probs = -F.nll_loss(F.softmax(logits, -1), targets, reduction="none")
        ranks = torch.where(
            logits.argsort(descending=True, dim=-1) == targets[..., None]
        )[1]
        tokens = tokenizer.convert_ids_to_tokens(targets)

        for j, (prob, rank, token) in enumerate(zip(probs, ranks, tokens)):
            outputs[f"prob/input_{i}/token_{j}/{token}"] = prob
            outputs[f"rank/input_{i}/token_{j}/{token}"] = (rank + 1,)

        outputs["loss"].append(loss)
        outputs["attack_success_rate"].append(attack_success_rate)
        outputs[f"prompt_{i}"] = prompt
        outputs[f"continuation_{i}"] = continuation

    outputs["loss"] = torch.stack(outputs["loss"]).mean()
    outputs["attack_success_rate"] = torch.stack(outputs["attack_success_rate"]).mean()

    return outputs
