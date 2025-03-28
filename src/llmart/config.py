#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from warnings import warn
from dataclasses import dataclass, asdict, field
from hydra.core.config_store import ConfigStore
from hydra.types import RunMode
from omegaconf import MISSING

cs = ConfigStore.instance()


@dataclass(kw_only=True)
class CoreConf:
    hydra: dict = field(
        default_factory=lambda: dict(
            # https://github.com/facebookresearch/hydra/issues/2262
            mode=RunMode.RUN,
            job=dict(chdir=True, name="llmart"),
            run=dict(
                dir="./outputs/${hydra:job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S.%f}"
            ),
            callbacks=dict(fixed_now={"_target_": "llmart.callbacks.HydraFixedNow"}),
        )
    )
    output_dir: str = "${hydra:runtime.output_dir}"
    experiment_name: str = "${hydra:job.name}"
    seed: int = 2024
    use_deterministic_algorithms: bool = False

    def asdict(self, flatten=False, ignore=["defaults", "hydra"]):
        d = asdict(self)
        for key in ignore:
            d.pop(key, None)
        if flatten:
            d = _flatten(d, ignore=ignore)
        return d


def _flatten(d: dict, ignore: list[str] | None = None, _parent_key: str = "") -> dict:
    """Flattens a nested dictionary into a single-level dictionary.

    Args:
        d: Dictionary to flatten.
        ignore: List of keys to ignore during flattening.

    Returns:
        Flattened dictionary with concatenated keys.
    """

    ignore = ignore or []
    flattened_d = {}
    for key, value in d.items():
        if key in ignore:
            continue

        key = f"{_parent_key}/{key}" if _parent_key else key
        if isinstance(value, dict):
            flattened_d.update(_flatten(value, _parent_key=key, ignore=ignore))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                flattened_d[f"{key}.{i}"] = item
        else:
            flattened_d[key] = value

    return flattened_d


# Response
@dataclass(kw_only=True)
class ResponseConf:
    replace_with: str | None = None


# Attacks
@dataclass(kw_only=True)
class AttackConf:
    prefix: str | int | None = MISSING
    suffix: str | int | None = MISSING

    # re.sub attack
    repl: str | int | None = MISSING
    pattern: str | None = MISSING

    # padding
    prefix_pad_left: str = ""
    prefix_pad_right: str = " "
    suffix_pad_left: str = " "
    suffix_pad_right: str = ""
    repl_pad_left: str = " "
    repl_pad_right: str = " "

    default_token: str = " !"

    dim: int = 0
    init: str | None = None


@dataclass(kw_only=True)
class NoAttackConf(AttackConf):
    suffix: str | int | None = None
    prefix: str | int | None = None
    repl: str | int | None = None
    pattern: str | None = None


@dataclass(kw_only=True)
class SuffixAttackConf(NoAttackConf):
    suffix: str | int | None = 20


@dataclass(kw_only=True)
class PrefixAttackConf(NoAttackConf):
    prefix: str | int | None = 20


@dataclass(kw_only=True)
class ReplWithAttackConf(NoAttackConf):
    repl: str | int | None = MISSING
    pattern: str | None = MISSING


cs.store(name="custom", group="attack", node=AttackConf)
cs.store(name="suffix", group="attack", node=SuffixAttackConf)
cs.store(name="prefix", group="attack", node=PrefixAttackConf)
cs.store(name="replwith", group="attack", node=ReplWithAttackConf)
cs.store(name="none", group="attack", node=NoAttackConf)


# Data
@dataclass(kw_only=True)
class DataConf:
    path: str = MISSING
    files: str | None = None
    trust_remote_code: bool = True

    shuffle: bool = False
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_minitrain: int | None = None
    subset: list[int] | None = None
    split_batches: bool | None = None


@dataclass(kw_only=True)
class BasicDataConf(DataConf):
    path: str = "basic"


@dataclass(kw_only=True)
class AdvBenchBehavior(DataConf):
    path: str = "advbench_behavior"
    files: str | None = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/refs/heads/main/data/advbench/harmful_behaviors.csv"
    )

    # Default to first sample
    subset: list[int] | None = field(default_factory=lambda: [0])


@dataclass(kw_only=True)
class AdvBenchJudge(DataConf):
    path: str = "advbench_judge"
    files: str | None = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/refs/heads/main/data/advbench/harmful_strings.csv"
    )

    # Default to first sample
    subset: list[int] | None = field(default_factory=lambda: [0])


cs.store(name="custom", group="data", node=DataConf)
cs.store(name="basic", group="data", node=BasicDataConf)
cs.store(name="advbench_behavior", group="data", node=AdvBenchBehavior)
cs.store(name="advbench_judge", group="data", node=AdvBenchJudge)


# Losses
@dataclass(kw_only=True)
class CausalLMLossConf:
    name: str = "model"
    reduction: str = "mean"

    # ranking
    margin: float | None = None
    rank: int | None = None

    # hardmax reduction
    hm_wrong: int = 1

    # mellowmax reduction
    mm_alpha: float = 1.0


@dataclass(kw_only=True)
class LogitCausalLMLossConf(CausalLMLossConf):
    name: str = "logit"


@dataclass(kw_only=True)
class CrossEntropyCausalLMLossConf(CausalLMLossConf):
    name: str = "xent"


@dataclass(kw_only=True)
class RankingCausalLMLossConf(CausalLMLossConf):
    name: str = "ranking"
    margin: float | None = 1e-3
    rank: int | None = 0


cs.store(name="model", group="loss", node=CausalLMLossConf)
cs.store(name="logit", group="loss", node=LogitCausalLMLossConf)
cs.store(name="xent", group="loss", node=CrossEntropyCausalLMLossConf)
cs.store(name="ranking", group="loss", node=RankingCausalLMLossConf)


# Optimizer
@dataclass(kw_only=True)
class OptimConf:
    name: str = MISSING

    # GCG
    negative_only: bool = False
    coord_randk: int = 0
    coord_topk: int = 256
    global_topk: int = 0
    n_tokens: int = 20
    n_swaps: int = 1024
    n_buffers: int = 1
    ignore_curr_marginals: bool = False

    # SGD ∩ Adam
    lr: float = 0.001
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    # Adam
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False


@dataclass(kw_only=True)
class GCGConf(OptimConf):
    name: str = "gcg"


@dataclass(kw_only=True)
class SGDConf(OptimConf):
    name: str = "sgd"


@dataclass(kw_only=True)
class AdamConf(OptimConf):
    name: str = "adam"


cs.store(name="gcg", group="optim", node=GCGConf)
cs.store(name="sgd", group="optim", node=SGDConf)
cs.store(name="adam", group="optim", node=AdamConf)


# Schedulers
@dataclass(kw_only=True)
class SchedulerConf:
    name: str = MISSING
    var_name: str = "n_tokens"

    # Exponential, Multistep
    gamma: float = 1.0
    # Constant, Plateau
    factor: float = 1.0

    # Linear
    start_factor: float = 0.5
    end_factor: float = 0.0
    total_iters: int = 20

    # Cosine
    eta_min: float = 0.2
    T_max: int = 500

    # Multistep
    milestones: list[int] = field(default_factory=lambda: [100, 1000])

    # Plateau
    # Multiply with [factor] on plateau after [patience] steps of
    # loss not dropping more than [threshold], on a [treshold_mode] scale
    # Stay within [min_value, max_value]
    patience: int = 1000
    threshold: float = 0.01
    threshold_mode: str = "abs"
    min_value: int = 1
    max_value: int = 1024

    def __post_init__(self):
        if self.var_name not in ["n_tokens", "n_swaps", "coord_topk"]:
            raise ValueError("Invalid variable selected for scheduling!")

        if self.threshold_mode not in ["abs", "rel"]:
            raise ValueError(
                "Invalid 'threshold_mode' specified for the change-on-plateau scheduler!"
            )


@dataclass(kw_only=True)
class ConstantLRConf(SchedulerConf):
    name: str = "constant"


@dataclass(kw_only=True)
class LinearLRConf(SchedulerConf):
    name: str = "linear"


@dataclass(kw_only=True)
class ExponentialLRConf(SchedulerConf):
    name: str = "exponential"


@dataclass(kw_only=True)
class CosineAnnealingLRConf(SchedulerConf):
    name: str = "cosine"


@dataclass(kw_only=True)
class MultiStepLRConf(SchedulerConf):
    name: str = "multistep"


@dataclass(kw_only=True)
class ReduceLRonPlateauConf(SchedulerConf):
    name: str = "plateau"


cs.store(name="custom", group="scheduler", node=SchedulerConf)
cs.store(name="constant", group="scheduler", node=ConstantLRConf)
cs.store(name="linear", group="scheduler", node=LinearLRConf)
cs.store(name="exponential", group="scheduler", node=ExponentialLRConf)
cs.store(name="cosine", group="scheduler", node=CosineAnnealingLRConf)
cs.store(name="multistep", group="scheduler", node=MultiStepLRConf)
cs.store(name="plateau", group="scheduler", node=ReduceLRonPlateauConf)


# Pipelines
@dataclass(kw_only=True)
class PipelineConf:
    task: str = "text-generation"
    name: str = MISSING
    revision: str = MISSING
    device: str | None = "cuda"
    device_map: str | None = None
    torch_dtype: str = "bfloat16"
    chat_template: str | None = None


@dataclass(kw_only=True)
class Llama3_8B_InstructConf(PipelineConf):
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    revision: str = "e1945c40cd546c78e41f1151f4db032b271faeaa"


@dataclass(kw_only=True)
class Llama3p1_8B_InstructConf(PipelineConf):
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    revision: str = "0e9e39f249a16976918f6564b8830bc894c89659"


@dataclass(kw_only=True)
class Llama3p1_70B_InstructConf(PipelineConf):
    name: str = "meta-llama/Llama-3.1-70B-Instruct"
    revision: str = "945c8663693130f8be2ee66210e062158b2a9693"


@dataclass(kw_only=True)
class Llama3p2_1B_InstructConf(PipelineConf):
    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    revision: str = "e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14"


@dataclass(kw_only=True)
class Llama3p2_11B_VisionConf(PipelineConf):
    name: str = "meta-llama/Llama-3.2-11B-Vision"
    revision: str = "b35f54f2124a51ba67ade0fb95d1715f0c3b98c7"


@dataclass(kw_only=True)
class LlamaGuard3_1BConf(PipelineConf):
    name: str = "meta-llama/Llama-Guard-3-1B"
    revision: str = "9e4f4b019bb3e964efa227180a64adc015856111"


@dataclass(kw_only=True)
class GraySwan_Llama3_8BrrConf(PipelineConf):
    name: str = "GraySwanAI/Llama-3-8B-Instruct-RR"
    revision: str = "d92f951d380d3489fb56b08c296376ea61cebef0"


@dataclass(kw_only=True)
class Deepseek_R1_Distill_Llama_8B(PipelineConf):
    name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    revision: str = "24ae87a9c340aa4207dd46509414c019998e0161"
    chat_template: str | None = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
    )


cs.store(name="custom", group="model", node=PipelineConf)
cs.store(name="llama3-8b-instruct", group="model", node=Llama3_8B_InstructConf)
cs.store(name="llama3.1-8b-instruct", group="model", node=Llama3p1_8B_InstructConf)
cs.store(name="llama3.1-70b-instruct", group="model", node=Llama3p1_70B_InstructConf)
cs.store(name="llama3.2-1b-instruct", group="model", node=Llama3p2_1B_InstructConf)
cs.store(name="llama3.2-11b-vision", group="model", node=Llama3p2_11B_VisionConf)
cs.store(name="llamaguard3-1b", group="model", node=LlamaGuard3_1BConf)
cs.store(name="llama3-8b-grayswan-rr", group="model", node=GraySwan_Llama3_8BrrConf)
cs.store(
    name="deepseek-r1-distill-llama-8b",
    group="model",
    node=Deepseek_R1_Distill_Llama_8B,
)


# LLMart
@dataclass(kw_only=True)
class LLMartConf(CoreConf):
    defaults: list = field(
        default_factory=lambda: [
            {"optim": "gcg"},
            {"loss": "model"},
            {"attack": "suffix"},
            {"scheduler": "linear"},
            {"data": "advbench_behavior"},
            {"model": MISSING},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
            "_self_",
        ]
    )

    steps: int = 500
    early_stop: bool = True
    val_every: int = 50
    save_every: int = 50

    model: PipelineConf
    attack: AttackConf
    response: ResponseConf = field(default_factory=ResponseConf)
    loss: CausalLMLossConf
    closure_loss: CausalLMLossConf | None = None
    data: DataConf
    optim: OptimConf
    scheduler: SchedulerConf

    # sampler + dataloader
    per_device_bs: int | str = 1
    bs: int = 1
    with_replacement: bool = True
    use_kv_cache: bool = False

    def __post_init__(self):
        if isinstance(self.per_device_bs, str) and self.per_device_bs != "auto":
            raise ValueError(
                f"Hardware batch size ({self.per_device_bs}) must be either a positive integer or 'auto'!"
            )

        if isinstance(self.per_device_bs, int) and self.bs > self.per_device_bs:
            assert (
                (self.bs % self.per_device_bs) == 0
            ), f"Hardware (micro) batch size ({self.per_device_bs}) must divide the batch size ({self.bs})!"
        elif isinstance(self.per_device_bs, int) and self.bs < self.per_device_bs:
            assert (
                (self.per_device_bs % self.bs) == 0
            ), f"Batch size ({self.bs}) must divide hardware (micro) batch size ({self.per_device_bs})!"

        if (
            self.attack.suffix is None
            and self.attack.prefix is None
            and self.attack.repl is None
            and self.attack.pattern is None
        ):
            if self.steps != 0:
                warn("Setting steps to 0 because attack is none!")
            self.steps = 0


cs.store(name="llmart", node=LLMartConf)
