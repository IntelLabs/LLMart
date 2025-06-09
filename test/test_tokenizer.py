#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import pytest
from transformers import logging, AutoTokenizer
from collections.abc import MutableMapping

from llmart import TaggedTokenizer

TOKENIZER_PATHS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Intel/neural-chat-7b-v3-3",
]
TOKENIZER_REVISIONS = {
    "Intel/neural-chat-7b-v3-3": "7506dfc5fb325a8a8e0c4f9a6a001671833e5b8e"
}

UNSUPPORTED_TOKENIZER_PATHS = []

logging.set_verbosity_error()


@pytest.fixture(scope="function", params=TOKENIZER_PATHS)
def tok(request):
    revision = TOKENIZER_REVISIONS.get(request.param, None)
    tok = AutoTokenizer.from_pretrained(
        request.param, revision=revision, local_files_only=True
    )
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def tag_tok(tok):
    return TaggedTokenizer(
        tok,
        tags=[
            "<|begin_prefix|>",
            "<|end_prefix|>",
            "<|begin_suffix|>",
            "<|end_suffix|>",
        ],
    )


def assert_inputs_equal(inputs1, inputs2):
    input1_ids = inputs1
    if isinstance(inputs1, MutableMapping):
        input1_ids = inputs1["input_ids"]

    input2_ids = inputs2
    if isinstance(inputs2, MutableMapping):
        input2_ids = inputs2["input_ids"]

    if isinstance(input1_ids, torch.Tensor):
        torch.testing.assert_close(input1_ids, input2_ids)
    elif isinstance(input1_ids[0], list):
        assert len(input1_ids) == len(input2_ids)
        for input1_id, input2_id in zip(input1_ids, input2_ids):
            assert_inputs_equal(input1_id, input2_id)
    else:
        assert input1_ids == input2_ids


@pytest.mark.parametrize("path", UNSUPPORTED_TOKENIZER_PATHS)
def test_bad_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)

    # These tokenizers are explicitly marked as bad in TaggedTokenizer
    with pytest.raises(ValueError):
        TaggedTokenizer(tok)


ATTACKS = [
    # same length
    ("@ @ @", "@ @ @"),
    # different lengths
    ("@ @ @", "@ @ @ @"),
    ("@ @ @ @", "@ @ @"),
    # space before/after
    ("@ @ @", " @ @ @"),
    (" @ @ @", "@ @ @"),
    (" @ @ @", " @ @ @"),
    ("@ @ @", "@ @ @ "),
    ("@ @ @ ", " @ @ @"),
    ("@ @ @ ", "@ @ @ "),
    ("@ @ @ ", "@ @ @"),
    (" @ @ @", "@ @ @ "),
]
PROMPT = "Hello"
COMPLETION = "World"


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer(tok, tag_tok, prefix, suffix):
    text = f"{prefix}{PROMPT}{suffix}{COMPLETION}"
    assert_inputs_equal(
        tok(text),
        tag_tok(text),
    )

    assert_inputs_equal(
        tok([text]),
        tag_tok([text]),
    )


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_pt(tok, tag_tok, prefix, suffix):
    text = f"{prefix}{PROMPT}{suffix}{COMPLETION}"
    assert_inputs_equal(
        tok(text, return_tensors="pt"),
        tag_tok(text, return_tensors="pt"),
    )

    assert_inputs_equal(
        tok([text], return_tensors="pt"),
        tag_tok([text], return_tensors="pt"),
    )


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_chat(tok, tag_tok, prefix, suffix):
    conv = [
        dict(role="user", content=f"{prefix}{PROMPT}{suffix}"),
        dict(role="assistant", content=COMPLETION),
    ]

    # since no adversarial tokens, these should match
    assert_inputs_equal(
        tok.apply_chat_template(conv),
        tag_tok.apply_chat_template(conv),
    )

    # since no adversarial tokens, these should match
    assert_inputs_equal(
        tok.apply_chat_template(conv, return_tensors="pt"),
        tag_tok.apply_chat_template(conv, return_tensors="pt"),
    )


def test_tokenizer_tag_fails(tag_tok):
    # Tokens in back-to-back adversarial blocks can be ambiguous
    # The example below turns into "Hello" which will tokenize into a single token!
    with pytest.raises(ValueError):
        tag_tok(
            "<|begin_prefix|>Hel<|end_prefix|><|begin_suffix|>lo<|end_suffix|>",
            return_tensors="pt",
        )


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_adv(tok, tag_tok, prefix, suffix):
    text = f"{prefix.strip()} {PROMPT} {suffix.strip()} {COMPLETION}"
    inputs = tok(text, return_tensors="pt")

    tag_text = f"{prefix} {PROMPT} {suffix} {COMPLETION}"
    tag_text = f"<|begin_prefix|>{prefix}<|end_prefix|> {PROMPT} <|begin_suffix|>{suffix}<|end_suffix|> {COMPLETION}"
    tag_inputs = tag_tok(tag_text, return_tensors="pt")

    # adversarial tokenizer should match non-adversarial tokenizer
    assert_inputs_equal(inputs, tag_inputs)

    # check prefix masked input ids match prefix ids
    assert "prefix_mask" in tag_inputs
    tag_prefix_ids = tag_inputs["input_ids"][tag_inputs["prefix_mask"]]
    assert tag_tok.decode(tag_prefix_ids).strip() == prefix.strip()

    # check suffix masked input ids match suffix ids
    assert "suffix_mask" in tag_inputs
    tag_suffix_ids = tag_inputs["input_ids"][tag_inputs["suffix_mask"]]
    assert tag_tok.decode(tag_suffix_ids).strip() == suffix.strip()


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_adv_nospecial(tok, tag_tok, prefix, suffix):
    text = f"{prefix.strip()} {PROMPT} {suffix.strip()} {COMPLETION}"
    inputs = tok(text, return_tensors="pt", add_special_tokens=False)

    tag_text = f"{prefix} {PROMPT} {suffix} {COMPLETION}"
    tag_text = f"<|begin_prefix|>{prefix}<|end_prefix|> {PROMPT} <|begin_suffix|>{suffix}<|end_suffix|> {COMPLETION}"
    tag_inputs = tag_tok(tag_text, return_tensors="pt", add_special_tokens=False)

    # adversarial tokenizer should match non-adversarial tokenizer
    assert_inputs_equal(inputs, tag_inputs)

    # check prefix masked input ids match prefix ids
    assert "prefix_mask" in tag_inputs
    tag_prefix_ids = tag_inputs["input_ids"][tag_inputs["prefix_mask"]]
    assert tag_tok.decode(tag_prefix_ids).strip() == prefix.strip()

    # check suffix masked input ids match suffix ids
    assert "suffix_mask" in tag_inputs
    tag_suffix_ids = tag_inputs["input_ids"][tag_inputs["suffix_mask"]]
    assert tag_tok.decode(tag_suffix_ids).strip() == suffix.strip()


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_tag_chat(tok, tag_tok, prefix, suffix):
    conv = [
        dict(role="user", content=f"{prefix.strip()} {PROMPT} {suffix.strip()}"),
        dict(role="assistant", content=COMPLETION),
    ]
    inputs = tok.apply_chat_template(conv, return_tensors="pt", return_dict=True)

    tag_conv = [
        dict(
            role="user",
            content=f"<|begin_prefix|>{prefix}<|end_prefix|> {PROMPT} <|begin_suffix|>{suffix}<|end_suffix|>",
        ),
        dict(
            role="assistant",
            content=COMPLETION,
        ),
    ]
    tag_inputs = tag_tok.apply_chat_template(
        tag_conv, return_tensors="pt", return_dict=True
    )

    # adversarial tokenizer should match non-adversarial tokenizer
    assert_inputs_equal(inputs, tag_inputs)

    # prefix masked input ids should match tokenized prefix
    assert "prefix_mask" in tag_inputs
    tag_prefix_ids = tag_inputs["input_ids"][tag_inputs["prefix_mask"]]
    assert tag_tok.decode(tag_prefix_ids).strip() == prefix.strip()

    # suffix masked input ids should match tokenized suffix
    assert "suffix_mask" in tag_inputs
    tag_suffix_ids = tag_inputs["input_ids"][tag_inputs["suffix_mask"]]
    assert tag_tok.decode(tag_suffix_ids).strip() == suffix.strip()


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_tag_chat_tiled(tok, tag_tok, prefix, suffix):
    del suffix

    conv = [
        dict(role="user", content=f"{prefix.strip()} {prefix.strip()} {PROMPT}"),
        dict(role="assistant", content=COMPLETION),
    ]
    inputs = tok.apply_chat_template(conv, return_tensors="pt", return_dict=True)

    tag_conv = [
        dict(
            role="user",
            content=f"<|begin_prefix|>{prefix}<|end_prefix|> <|begin_prefix|>{prefix}<|end_prefix|> {PROMPT}",
        ),
        dict(
            role="assistant",
            content=COMPLETION,
        ),
    ]
    tag_inputs = tag_tok.apply_chat_template(
        tag_conv, return_tensors="pt", return_dict=True
    )

    # adversarial tokenizer should match non-adversarial tokenizer
    assert_inputs_equal(inputs, tag_inputs)

    # prefix masked input ids should match tokenized prefix
    assert "prefix_mask" in tag_inputs
    tag_prefix_ids = tag_inputs["input_ids"][tag_inputs["prefix_mask"]]
    assert (
        tag_tok.decode(tag_prefix_ids).strip() == f"{prefix.strip()} {prefix.strip()}"
    )

    # there should be no suffix mask
    assert tag_inputs["suffix_mask"].any().item() is False


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_tag_chat_prompt_like_suffix(tok, tag_tok, prefix, suffix):
    del prefix

    conv = [
        dict(role="user", content=f"{suffix} {PROMPT} {suffix.strip()}"),
        dict(role="assistant", content=COMPLETION),
    ]
    inputs = tok.apply_chat_template(conv, return_tensors="pt", return_dict=True)

    tag_conv = [
        dict(
            role="user",
            content=f"{suffix} {PROMPT} <|begin_suffix|>{suffix}<|end_suffix|>",
        ),
        dict(
            role="assistant",
            content=COMPLETION,
        ),
    ]
    tag_inputs = tag_tok.apply_chat_template(
        tag_conv, return_tensors="pt", return_dict=True
    )

    # adversarial tokenizer should match non-adversarial tokenizer
    assert_inputs_equal(inputs, tag_inputs)

    # there should be no prefix mask
    assert tag_inputs["prefix_mask"].any().item() is False

    # suffix masked input ids should match tokenized suffix
    assert "suffix_mask" in tag_inputs
    tag_suffix_ids = tag_inputs["input_ids"][tag_inputs["suffix_mask"]]
    assert tag_tok.decode(tag_suffix_ids).strip() == suffix.strip()

    # convert prompt tokens into list of individual stripped tokens since
    # we need to find the PROMPT token
    prompt_tokens = [
        tok.strip() for tok in tag_tok.batch_decode(tag_inputs["input_ids"][0])
    ]
    # suffix mask should come after PROMPT (and not before!!)
    prompt_loc = prompt_tokens.index(PROMPT)
    suffix_locs = torch.where(tag_inputs["suffix_mask"][0])[0]
    assert len(suffix_locs) == len(tag_suffix_ids)
    assert (suffix_locs > prompt_loc).all()


@pytest.mark.parametrize("prefix,suffix", ATTACKS)
def test_tokenizer_tag_chat_open(tag_tok, prefix, suffix):
    del prefix

    tag_conv = [
        dict(
            role="user",
            content=f"{PROMPT} <|begin_suffix|>{suffix}",
        ),
        dict(
            role="assistant",
            content=COMPLETION,
        ),
    ]

    with pytest.raises(ValueError):
        tag_tok.apply_chat_template(tag_conv, return_tensors="pt", return_dict=True)


@pytest.mark.parametrize(
    "conv",
    [
        [dict(role="user", content="")],
        [dict(role="user", content="hello")],
        [
            dict(role="user", content="hello"),
            dict(role="assistant", content=""),
        ],
        [
            dict(role="user", content="hello"),
            dict(role="assistant", content="world"),
        ],
    ],
)
def test_tokenizer_reencodes(tag_tok, conv):
    input_ids: torch.Tensor = tag_tok.apply_chat_template(conv, return_tensors="pt")  # type: ignore
    assert tag_tok.reencodes(input_ids)


@pytest.mark.skip
def test_tokenizer_chat_template(tok):
    prompt_ids = tok.apply_chat_template(
        [
            dict(role="user", content="prompt"),
        ],
        add_generation_prompt=True,
    )

    empty_assistant_ids = tok.apply_chat_template(
        [
            dict(role="user", content="prompt"),
            dict(role="assistant", content=""),
        ],
        add_generation_prompt=False,
    )

    completion_ids = tok.apply_chat_template(
        [
            dict(role="user", content="prompt"),
            dict(role="assistant", content="completion"),
        ],
        add_generation_prompt=False,
    )

    assert prompt_ids[-1] != tok.eos_token_id
    assert empty_assistant_ids[-1] == tok.eos_token_id
    assert completion_ids[-1] == tok.eos_token_id
