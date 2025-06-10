#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import copy
import math
import signal
import threading
import platform
import logging
import torch
import datasets
import itertools
import transformers
from typing import Callable
from functools import partial
from collections import defaultdict, OrderedDict
from accelerate import Accelerator, PartialState, find_executable_batch_size
from accelerate.logging import get_logger
from accelerate.utils import reduce, tqdm, DataLoaderConfiguration, set_seed
from accelerate.utils import is_cuda_available, is_xpu_available
from transformers import PreTrainedModel, pipeline, default_data_collator
from transformers import PreTrainedTokenizerFast, AutoProcessor, ProcessorMixin
from transformers.generation.utils import ModelOutput
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F

from llmart import config, data, optim, transforms, losses, schedulers
from llmart import TaggedTokenizer, AdversarialAttack, AttackPrompt

# Device helpers for efficient verbose memory usage tracking
_is_cuda_available = is_cuda_available()
_is_xpu_available = is_xpu_available()


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
    set_seed(cfg.seed, deterministic=cfg.use_deterministic_algorithms)

    # Instantiate world and accelerator
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=cfg.output_dir,
        dataloader_config=DataLoaderConfiguration(
            split_batches=cfg.data.split_batches  # type: ignore[reportArgumentType]
            if cfg.data.split_batches is not None
            else cfg.data.n_train > 1,
            use_seedable_sampler=(cfg.data.n_train == 1),
        ),
        step_scheduler_with_optimizer=False,
    )
    accelerator.init_trackers(cfg.experiment_name, config=cfg.asdict(flatten=True))

    # Setup logging
    if not accelerator.is_main_process:
        transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
    datasets.utils.disable_progress_bars()
    log = get_logger(__name__)  # type: ignore[reportArgumentType]
    log.info(f"{cfg.output_dir=}")

    # Create attack and responses dataset transforms
    modify_prompt = transforms.from_config(cfg.attack)
    force_completion = transforms.from_config(cfg.response)
    assert isinstance(modify_prompt, AttackPrompt)

    # Create adversarial processor (or tokenizer if it doesn't exist)
    processor = AutoProcessor.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        trust_remote_code=True,
        use_fast=True,
        **cfg.model.processor_kwargs,
    )
    if isinstance(processor, ProcessorMixin):
        tokenizer = getattr(processor, "tokenizer")
    elif isinstance(processor, PreTrainedTokenizerFast):
        tokenizer = processor
        processor = None
    else:
        raise NotImplementedError(f"Unknown processor: {processor}")

    tokenizer.chat_template = cfg.model.chat_template or tokenizer.chat_template
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer = TaggedTokenizer(
        tokenizer,
        tags=modify_prompt.tags + force_completion.tags,
        banned_strings=cfg.banned_strings,
    )
    if isinstance(processor, ProcessorMixin):
        setattr(processor, "tokenizer", tokenizer)

    # Create data, apply attack transforms to it
    with accelerator.main_process_first():
        ds = data.from_config(
            cfg.data,
            tokenizer=tokenizer,
            processor=processor,
            modify_prompt=modify_prompt,
            force_completion=force_completion,
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
        processor=processor,
        model_kwargs=dict(),
    )
    model = pipe.model
    model.requires_grad_(False)

    # Optimize attack
    step, attack = 0, None
    best_step, best_attack = 0, None
    results = dict()
    if len(modify_prompt.elements) > 0:
        if cfg.per_device_bs == -1:
            _train = find_executable_batch_size(train)
        else:
            _train = partial(train, cfg.per_device_bs)
        step, attack, best_step, best_attack, train_results = _train(
            ds,
            modify_prompt,
            tokenizer,  # type: ignore
            model,
            cfg,
            accelerator,
            log,
        )
        results.update(train_results)

    # Evaluate test data on final and best steps
    test_dl = DataLoader(ds["test"], collate_fn=default_data_collator)  # type: ignore
    if len(test_dl):
        log.info(f"== TEST @ {step} ==")
        outputs = evaluate(test_dl, tokenizer, model, attack, log, cfg.max_new_tokens)
        outputs = {f"eval/test_{key}": value for key, value in outputs.items()}
        results.update(outputs)
        accelerator.log(outputs, step=step)

        log.info(f"== TEST @ BEST {best_step} ==")
        outputs = evaluate(
            test_dl, tokenizer, model, best_attack, log, cfg.max_new_tokens
        )
        outputs = {f"eval/best_{key}": value for key, value in outputs.items()}
        results.update(outputs)
        accelerator.log(outputs, step=best_step)

    accelerator.end_training()

    return results


def train(
    per_device_bs: int,
    ds: datasets.DatasetDict,
    modify_prompt: AttackPrompt,
    tokenizer: TaggedTokenizer,
    model: PreTrainedModel,
    cfg: config.LLMartConf,
    accelerator: Accelerator,
    log: logging.Logger | logging.LoggerAdapter,
) -> tuple[int, AdversarialAttack, int, AdversarialAttack, dict]:
    if cfg.per_device_bs == -1:
        log.info(f"Trying {per_device_bs=}")
    global TRAIN_STOP
    TRAIN_STOP = False

    def signal_handler(signum, frame):
        log.info("Exiting early because of SIGUSR1!")
        global TRAIN_STOP
        TRAIN_STOP = True

    if (
        platform.system() == "Linux"
        and threading.current_thread() is threading.main_thread()
    ):
        signal.signal(signal.SIGUSR1, signal_handler)

    # Create adversarial attack and losses from tokenized prompt attack
    attack_inits = tokenizer(
        modify_prompt.elements,
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
        return 0, attack, 0, attack, dict()

    # Dataloaders
    train_bs = cfg.bs
    # For data parallelism, we attempt to make the data batch size divisible by the number of processes if needed and possible
    if accelerator.split_batches and train_bs % accelerator.num_processes != 0:
        new_train_bs = (
            math.ceil(train_bs / accelerator.num_processes) * accelerator.num_processes
        )
        # Don't spill over dataset size
        if new_train_bs > len(ds["train"]):
            new_train_bs = min(
                accelerator.num_processes, new_train_bs - accelerator.num_processes
            )
            # Can't have zero batch size, instead raise error
            if new_train_bs == 0:
                raise ValueError(
                    f"User requested {cfg.bs = }, but no suitable value can be accommodated for the given dataset and number of devices!"
                )

        log.warning(
            f"Training batch size {train_bs} is not divisible by {accelerator.num_processes = }. "
            f"Using training batch size {new_train_bs} instead."
        )
        train_bs = new_train_bs

    eval_dl = DataLoader(ds["train"], collate_fn=default_data_collator)  # type: ignore
    train_dl = DataLoader(
        ds["train"],  # type: ignore
        collate_fn=default_data_collator,
        batch_size=train_bs,
        sampler=RandomSampler(
            ds["train"],
            # No replacement for full-batch GD
            replacement=cfg.with_replacement if train_bs < len(ds["train"]) else False,
            num_samples=train_bs * cfg.steps
            if accelerator.split_batches
            else train_bs * cfg.steps * accelerator.num_processes,
        ),
    )
    minitrain_dl = DataLoader(ds["minitrain"], collate_fn=default_data_collator)  # type: ignore
    val_dl = DataLoader(ds["val"], collate_fn=default_data_collator)  # type: ignore

    # Create optimizer from adversarial attack parameters
    loss_fn = losses.from_config(cfg.loss)
    optimizer = optim.from_config(
        cfg.optim,
        attack.parameters(),
        good_token_ids=tokenizer.good_token_ids,
        embedding=attack.embedding if cfg.attack.dim == 1 else None,
    )
    scheduler = schedulers.from_config(
        cfg.scheduler,
        optimizer,
    )

    train_dl, model, optimizer, attack, scheduler = accelerator.prepare(
        train_dl, model, optimizer, attack, scheduler
    )

    # Make closure to pass to optimizer
    closure, closure_inputs = make_closure(
        attack,
        model,
        losses.from_config(cfg.closure_loss or cfg.loss),
        is_valid_input=tokenizer.reencodes,
        num_samples=train_bs,
        batch_size=per_device_bs,
        use_kv_cache=cfg.use_kv_cache,
        ignored_keys=tokenizer.mask_names,
    )

    # For each optimization step
    step, results = 0, dict()
    best_step, best_success_rate, best_attack = 0, 0, copy.deepcopy(attack)

    for step, inputs in (
        pbar := tqdm(iterable=enumerate(train_dl), total=len(train_dl), desc="steps")
    ):
        optimizer.zero_grad()

        model_loss, loss, attack_success, attack_count = 0.0, 0.0, 0, 0
        for micro_inputs in data.microbatch(inputs, micro_batch_size=per_device_bs):
            # Get adversarial version of inputs and pop adversarial tags
            micro_inputs = attack(micro_inputs)
            micro_inputs = {
                k: v for k, v in micro_inputs.items() if k not in tokenizer.mask_names
            }

            # Pop input_ids since we do not want to pass them to the model
            if not tokenizer.reencodes(
                micro_inputs.pop("input_ids"), micro_inputs["attention_mask"]
            ).all():
                log.warning("Adversarial inputs do not reencode.")

            # Compute loss using input_embeds
            outputs = model(**micro_inputs, use_cache=False)
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
                attack_success = reduce(attack_success, reduction="sum")
                attack_count = reduce(attack_count, reduction="sum")
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
            if _is_cuda_available:
                postfix["cuda_mem"] = (
                    f"{torch.cuda.max_memory_allocated() / (1024**2):0.3f}MiB"
                )
            if _is_xpu_available:
                postfix["xpu_mem"] = (
                    f"{torch.xpu.max_memory_allocated() / (1024**2):0.3f}MiB"
                )
            pbar.set_postfix(postfix)

            # Save tokens with highest success rate
            if success_rate >= best_success_rate:
                best_step = step
                best_success_rate, best_attack = success_rate, copy.deepcopy(attack)

            # Exit attack loop if we found a successful attack across all training examples
            if (
                cfg.early_stop and torch.allclose(success_rate, torch.tensor(1.0))  # type: ignore[reportArgumentType]
            ):
                # NOTE: We use evaluate because model() can differ from model.generate()
                outputs = evaluate(eval_dl, tokenizer, model, attack, max_new_tokens=0)
                if torch.allclose(outputs["attack_success_rate"], torch.tensor(1.0)):
                    break
            if TRAIN_STOP:
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
                results.update(outputs)
                accelerator.log(outputs, step=step)
            if len(val_dl) and cfg.val_every and step % cfg.val_every == 0:
                log.info(f"== VAL @ {step} ==")
                outputs = evaluate(val_dl, tokenizer, model, attack, log)
                outputs = {f"val/{key}": value for key, value in outputs.items()}
                results.update(outputs)
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

        best_attack_path = f"{cfg.output_dir}/best_attack_{best_step}.pt"
        torch.save(accelerator.unwrap_model(best_attack).state_dict(), best_attack_path)
        log.info(f"{best_attack_path=}")

    return step, attack, best_step, best_attack, results


def make_closure(
    attack: torch.nn.Module,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    is_valid_input: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_samples: int = 1,
    batch_size: int = 1,
    use_kv_cache: bool = False,
    ignored_keys: list[str] | None = None,
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
        ignored_keys: Names of keys to remove from inputs before passing to model

    Returns:
        A closure/generator and closure inputs to update before passing closure.
    """
    inputs = {}
    ignored_keys = ignored_keys or []

    def generator():
        """A generator that accumulates attacks on **a single training example** until
        a desired batch size, and then computes per-attack losses. No loss is computed
        for non-valid attacks.

        Yields:
            List of tuples containing attack indices and their losses.
        """

        param_losses = []
        batch = defaultdict(list)
        kv_cache, inputs_cache = None, None
        kv_cache_len = 0

        while True:
            # Get next attack
            param_idx = yield param_losses

            # If we have a whole batch, or a partial batch and we're stopping,
            # then compute per-example losses
            if (len(batch["param_idx"]) == batch_size) or (
                param_idx is None and len(batch["param_idx"])
            ):
                batch_param_idx = batch.pop("param_idx")

                # Construct batch by concatenating tensors along batch axis
                batch = {k: torch.cat(v) for k, v in batch.items()}
                if kv_cache is not None:
                    batch_kv_cache = copy.deepcopy(kv_cache)
                    batch_kv_cache.batch_repeat_interleave(len(batch_param_idx))
                    batch["past_key_values"] = batch_kv_cache

                outputs = model(**batch)

                # Compute per-example loss
                losses = itertools.starmap(
                    loss_fn, zip(outputs["logits"], batch["labels"])
                )
                param_losses = list(zip(batch_param_idx, losses))
                del outputs, losses

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
            input_map = adv_inputs.pop("input_map")
            adv_inputs = {k: v for k, v in adv_inputs.items() if k not in ignored_keys}
            if not is_valid_input(
                adv_inputs["input_ids"], adv_inputs["attention_mask"]
            ).all():
                del adv_inputs
                continue

            # ...compute past key values...
            if use_kv_cache and kv_cache is None:
                # Truncate all inputs to cache length by finding first non-zero input map token
                kv_cache_len = input_map.nonzero()[0, 1].item()
                cache_inputs = {
                    k: v[:, :kv_cache_len]
                    if len(v.shape) > 1 and v.shape[1] == input_map.shape[1]
                    else v
                    for k, v in adv_inputs.items()
                }
                outputs = model(**cache_inputs, use_cache=True)
                kv_cache = outputs["past_key_values"]
                del outputs

            # ...verify that inputs match the cache...
            if use_kv_cache and inputs_cache is not None:
                assert inputs_cache.equal(adv_inputs["input_ids"][:, :kv_cache_len]), (
                    "Input ids do not match the cached ones!"
                )

            # ...and if they do accumulate a batch
            batch["param_idx"].append(param_idx)
            for key, value in adv_inputs.items():
                # Truncate inputs that match kv-cache shape, ignoring attention_mask since
                # models always need a full attention mask.
                if (
                    key not in ("attention_mask",)
                    and len(value.shape) > 1
                    and value.shape[1] == input_map.shape[1]
                ):
                    value = value[:, kv_cache_len:]
                batch[key].append(value)
            del adv_inputs, input_map

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
            adv_inputs = {k: v for k, v in adv_inputs.items() if k not in ignored_keys}
            if not is_valid_input(
                adv_inputs["input_ids"], adv_inputs["attention_mask"]
            ).all():
                loss = torch.tensor(torch.inf, device=adv_inputs["input_ids"].device)
                break
            else:
                outputs = model(**adv_inputs, use_cache=False)
                micro_loss = loss_fn(outputs, micro_inputs["labels"])
                # Accumulate averages across micro-batches
                # NOTE: Assumes equal distribution of micro-batch across devices
                loss = loss + micro_loss * len(micro_inputs["labels"])

        # Average across the entire training batch
        loss = loss / len(inputs["labels"])

        return loss

    return (generator if num_samples == 1 else closure), inputs


@torch.random.fork_rng(
    devices=range(PartialState().num_processes), device_type=str(PartialState().device)
)
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
    outputs["loss"], outputs["loss_tf"] = [], []
    outputs["attack_success_rate"] = []

    for i, inputs in enumerate(dataloader):
        assert len(inputs["input_ids"]) == 1
        # Remove non-attended items
        attention_mask = inputs["attention_mask"]
        inputs = {
            k: v[attention_mask == 1][None] if v.shape == attention_mask.shape else v
            for k, v in inputs.items()
        }
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs = attack(inputs) if attack else inputs
        response_mask = inputs.pop("response_mask")
        inputs = {k: v for k, v in inputs.items() if k not in tokenizer.mask_names}

        # Measure teacher forcing loss
        output_tf = model(**inputs)
        loss_tf = output_tf["loss"]

        # Truncate all inputs to prompt length by finding first response index
        prompt_end = response_mask.nonzero()[0, 1].item()
        gen_inputs = {
            k: v[:, :prompt_end]
            if len(v.shape) > 1 and v.shape[1] == response_mask.shape[1]
            else v
            for k, v in inputs.items()
        }

        # Decode prompt
        prompt = tokenizer.decode(gen_inputs["input_ids"][0])
        log.info(f"{prompt=}") if log else None

        # Deterministically generate a response using prompt_ids
        output = model.generate(  # type: ignore[reportCallIssue]
            **gen_inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=None,
            max_new_tokens=max(max_new_tokens, response_mask.shape[1] - prompt_end),
            return_dict_in_generate=True,
            output_logits=True,
            return_legacy_cache=False,
        )

        # Decode continuation of prompt
        continuation_ids = output.sequences[0, prompt_end:]  # type: ignore
        continuation = tokenizer.decode(continuation_ids)

        # Compute loss
        targets = inputs["input_ids"][0, prompt_end:]
        continuation_mask = response_mask[0, prompt_end:]

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
        outputs["loss_tf"].append(loss_tf)
        outputs["attack_success_rate"].append(attack_success_rate)
        outputs[f"prompt_{i}"] = prompt
        outputs[f"continuation_{i}"] = continuation

    outputs["loss"] = torch.stack(outputs["loss"]).mean()
    outputs["loss_tf"] = torch.stack(outputs["loss_tf"]).mean()
    outputs["attack_success_rate"] = torch.stack(outputs["attack_success_rate"]).mean()

    return outputs
