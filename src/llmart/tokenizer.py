#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import re
import torch
from functools import cached_property
from transformers import PreTrainedTokenizerFast, BatchEncoding, AddedToken

BEGIN_TAG_FORMAT = "<|begin_%s|>"
BEGIN_TAG_PATTERN = r"<\|begin_(?P<name>\w+)\|>"

END_TAG_FORMAT = "<|end_%s|>"
END_TAG_PATTERN = r"<\|end_(?P<name>\w+)\|>"

MASK_FORMAT = "%s_mask"
MASK_PATTERN = r"(?P<name>\w+)_mask"


# FIXME: Add support for return_assistant_tokens_mask for completions and chats
class TaggedTokenizer(PreTrainedTokenizerFast):
    """
    A tagged tokenizer enables HTML-like tags to be inserted into text and tokenized into appropriate masks.

    For example, a prompt like:
        This is a <|begin_tag|>tagged prompt<|end_tag|> with <|begin_tag2|>two tags<|end_tag2|>

    Will tokenize as if the tags where not inserted into the text
        This is a tagged prompt with two tags

    But will produce two masks (note their names match the tag names):
        tag_mask = [False, False, True, True, False, False, False]
        tag2_mask = [False, False, False, False, False, True, True]

    You can have multiple tags of the same name but it is your job to disambiguate them.
    Whitespace to the right of a begin tag and to the left of an end tag will be stripped.

    Args:
        tokenizer: Base PreTrainedTokenizerFast instance to extend
        tags: List of tag strings to recognize (e.g. ["<|begin_tag|>", "<|end_tag|>"])
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        tags: list[str] | None = None,
        bad_string_list: list[str] = [],
    ):
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        # Magically subclass tokenizer
        self.__class__ = type(
            tokenizer.__class__.__name__, (self.__class__, tokenizer.__class__), {}
        )
        self.__dict__ = tokenizer.__dict__
        self.__vocab_size = len(self)

        # Add tag tokens to tokenizer paying special attention to stripping
        # "<|tag|> word" -> "<|tag|>word"
        # "word <|tag|>" -> "word<|tag|>"
        self.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(
                        tag,
                        special=True,
                        rstrip=re.match(BEGIN_TAG_PATTERN, tag) is not None,
                        lstrip=re.match(END_TAG_PATTERN, tag) is not None,
                    )
                    for tag in tags or []
                ]  # type: ignore
            },
            replace_additional_special_tokens=False,
        )
        self.tags = tags or []
        self.bad_string_list = bad_string_list

        # Detect add_prefix_space
        self.add_prefix_space = (
            super().decode(super().encode("@", add_special_tokens=True))
            == f"{self.bos_token} @"
        )

    @property
    def tag_ids(self) -> list[int]:
        return self.convert_tokens_to_ids(self.tags)  # type: ignore

    def apply_chat_template(self, conversation, *args, **kwargs):
        """Applies the chat template while preserving tag information.

        See :func:`~transformers.PreTrainedTokenizerBase.apply_chat_template` for more info.
        """

        tokenize = kwargs.pop("tokenize", True)

        # Render conversation with tag tokens
        tagged_text = super().apply_chat_template(conversation, tokenize=False)

        # Remove tags from conversation and render it
        for tag in self.tags:
            conversation = self._remove_tag(tag, conversation)
        return super().apply_chat_template(
            conversation,  # type: ignore
            *args,
            tokenizer_kwargs={"tagged_text": tagged_text},
            tokenize=tokenize,
            **kwargs,
        )

    def __call__(
        self, text=None, *args, tagged_text: str | None = None, **kwargs
    ) -> BatchEncoding:
        """Tokenizes text while tracking tagged portions.

        See :func:`~transformers.PreTrainedTokenizerBase.__call__` for more info.
        """

        if text is None:
            raise ValueError("{self.__class__.__name__} only supports text as input")

        # Keep tagged text
        tagged_text = tagged_text or text

        # Remove tags from tagged text
        for tag in self.tags:
            text = self._remove_tag(tag, text)

        # Call original tokenizer and exit early if no tags
        inputs = super().__call__(text, *args, **kwargs)  # type: ignore
        if tagged_text == text:
            return inputs

        # Otherwise, tokenize tagged text...
        tagged_inputs = super().__call__(tagged_text, *args, **kwargs)  # type: ignore
        inputs_ids: torch.Tensor = inputs["input_ids"]  # type: ignore
        tagged_inputs_ids: torch.Tensor = tagged_inputs["input_ids"]  # type: ignore

        if not isinstance(inputs_ids, torch.Tensor) or not isinstance(
            tagged_inputs_ids, torch.Tensor
        ):
            raise ValueError('You must pass return_tensors="pt"')

        # ...and create a map of the input.
        inputs["input_map"] = self._create_inputs_map(inputs_ids, tagged_inputs_ids)

        # Turn input map into a series of boolean masks
        for tag_id, tag in zip(self.tag_ids, self.tags):
            if m := re.match(BEGIN_TAG_PATTERN, tag):
                inputs[MASK_FORMAT % m["name"]] = inputs["input_map"] == tag_id

        return inputs

    def _remove_tag(self, tag: str, text: list | dict | str) -> list | dict | str:
        """Removes a specific tag from text while preserving content.

        Args:
            tag: Tag to remove
            text: Input text, list, or dict to process

        Returns:
            Text with specified tag removed
        """

        if isinstance(text, list):
            return [self._remove_tag(tag, t) for t in text]

        elif isinstance(text, dict):
            return dict(
                role=text["role"],
                content=self._remove_tag(tag, text["content"]),
            )

        elif isinstance(text, str):
            tag_id: int = self.convert_tokens_to_ids(tag)  # type: ignore

            # escape token because it may have special RE characters
            tag = re.escape(tag)

            # make sure to strip left/right spaces too
            token = self.added_tokens_decoder[tag_id]
            if token.lstrip:
                tag = f"\\s*{tag}"
            if token.rstrip:
                tag = f"{tag}\\s*"

            # remove token from conversation
            return re.sub(tag, "", text)

    def _create_inputs_map(
        self, inputs_ids: torch.Tensor, tagged_inputs_ids: torch.Tensor
    ) -> torch.Tensor:
        """Creates mapping between tagged and untagged token sequences.

        Args:
            inputs_ids: Original token IDs
            tagged_inputs_ids: Token IDs with tags

        Returns:
            Tensor mapping of input tokens to their corresponding tags

        Raises:
            ValueError: If tagged elements are back-to-back or content not found
        """

        inputs_map = inputs_ids.clone().fill_(0)

        for input_ids, tagged_input_ids, input_map in zip(
            inputs_ids, tagged_inputs_ids, inputs_map
        ):
            first_iteration = True  # FIXME: I don't like this
            while True:
                # Remove common prefix since we can safely ignore these tokens since they
                # contain no tags.
                i = self._common_prefix_length(input_ids, tagged_input_ids)
                input_ids = input_ids[i:]
                input_map = input_map[i:]
                tagged_input_ids = tagged_input_ids[i:]

                span = self._get_next_element_span(tagged_input_ids)
                if span is None:
                    break

                start, stop = span

                # if we did not remove any common prefix and the next element span starts at the beginning
                # then elements must be back-to-back (except in the first iteration).
                if not first_iteration and i == 0 and start == 0:
                    raise ValueError(
                        "Tagged elements cannot be back-to-back since this induces ambiguous tokenizations."
                    )

                tag_id = tagged_input_ids[start]
                content = tagged_input_ids[start + 1 : stop]

                # Find location of content in inputs
                i = self._index_of(content, in_seq=input_ids)

                # Special case for tokenizers that add spaces to beginning of tokens. We try to find a match
                # for the first <token> using <space><token>.
                content_with_space: torch.Tensor = self.encode(  # type: ignore
                    " " + self.decode(content, clean_up_tokenization_spaces=False),
                    add_special_tokens=False,
                    return_tensors="pt",
                )[0]
                # use last len(content) tokens since we potentially added tokens
                i_space = self._index_of(
                    content_with_space[-len(content) :], in_seq=input_ids
                )
                # use left-most match index
                i = min(i, i_space)

                # Special case for when we don't find anything. We try to find a match for the first <token>
                # using <newline><token>.
                content_with_newline: torch.Tensor = self.encode(  # type: ignore
                    "\n" + self.decode(content, clean_up_tokenization_spaces=False),
                    add_special_tokens=False,
                    return_tensors="pt",
                )[0]
                # use last len(content) tokens since we potentially added tokens
                i_newline = self._index_of(
                    content_with_newline[-len(content) :],
                    in_seq=input_ids,
                )
                # use left-most match index
                i = min(i, i_newline)

                # If we didn't find the content in inputs, then show a nice error message.
                if i == len(input_ids):
                    content_toks = self.convert_ids_to_tokens(content)  # type: ignore
                    input_toks = self.convert_ids_to_tokens(input_ids)  # type: ignore
                    raise ValueError(f"Unable to find {content_toks} in {input_toks}!")

                # Set input map to tag id
                input_map[i : i + len(content)] = tag_id

                # Skip element
                input_map = input_map[i + len(content) :]
                input_ids = input_ids[i + len(content) :]
                tagged_input_ids = tagged_input_ids[stop + 1 :]

                first_iteration = False

        return inputs_map

    def _common_prefix_length(self, seq1: torch.Tensor, seq2: torch.Tensor) -> int:
        """Finds length of common prefix between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Length of common prefix
        """

        for i, (id1, id2) in enumerate(zip(seq1, seq2)):
            if id1 != id2:
                return i
        return min(len(seq1), len(seq2))

    def _index_of(self, seq: torch.Tensor, in_seq: torch.Tensor) -> int:
        """Finds first occurrence of subsequence in sequence.

        Args:
            seq: Subsequence to find
            in_seq: Sequence to search in

        Returns:
            Starting index of match, or len(in_seq) if not found
        """

        for i in range(len(in_seq) - len(seq) + 1):
            if torch.all(in_seq[i : i + len(seq)] == seq):
                return i
        return len(in_seq)

    def _get_next_element_span(
        self,
        ids: torch.Tensor,
    ) -> tuple[int, int] | None:
        """Finds span of next tagged element in token sequence.

        Args:
            ids: Sequence of token IDs

        Returns:
            Tuple of (start, end) indices, or None if no element found

        Raises:
            ValueError: If tags are mismatched or malformed
        """

        start = None
        begin_token = None

        for i, (token_id, token) in enumerate(
            zip(ids, self.convert_ids_to_tokens(ids))  # type: ignore
        ):
            # If we have a begin tag...
            if token_id in self.tag_ids and (m := re.match(BEGIN_TAG_PATTERN, token)):
                if start is not None:
                    raise ValueError("double begin tags detected!")

                # keep track of begin tag
                start = i
                begin_token = m["name"]

            # If we have an end tag, then yield contents of element
            if token_id in self.tag_ids and (m := re.match(END_TAG_PATTERN, token)):
                if start is None:
                    raise ValueError("end tag with no begin tag!")
                if m["name"] != begin_token:
                    raise ValueError("begin/end tags are mismatched!")

                return start, i

        if start is not None:
            raise ValueError("No end tag found!")

        return None

    def decode(self, *args, **kwargs):
        """Decodes token IDs to text while handling special tokens.

        See :func:`~transformers.PreTrainedTokenizerBase.decode` for more info.
        """

        decoded = super().decode(*args, **kwargs)

        # NOTE: For some reason Llama2 tokenizers adds a space to special tokens,
        #       so we add special cases here.
        #       See: https://github.com/huggingface/tokenizers/issues/1237
        if self.add_prefix_space:
            for special_token in self.all_special_tokens:
                decoded = decoded.replace(f"{special_token} ", f"{special_token}")

        return decoded

    def reencodes(self, sequences: torch.Tensor) -> torch.Tensor:
        """Tests if sequences can be perfectly reconstructed after decoding/encoding.

        Args:
            sequences: Token sequences to test

        Returns:
            Boolean tensor indicating which sequences reencoded perfectly
        """

        device = sequences.device

        assert len(sequences.shape) == 2
        sequences = sequences.cpu()

        sentences = self.batch_decode(sequences)

        # Pad to same length as original sequence
        new_sequences = self(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=sequences.shape[-1],
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

        return torch.eq(sequences, new_sequences).all(-1).to(device=device)

    @cached_property
    def good_token_ids(self) -> torch.Tensor:
        added_tokens = self.added_tokens_encoder.keys()
        tokens = [
            self.convert_tokens_to_string([token])
            for token in self.convert_ids_to_tokens(list(range(self.__vocab_size)))
        ]
        printable_tokens = torch.tensor(
            [
                token.isprintable()
                and token.isascii()
                and token not in added_tokens
                and len(token.strip()) > 0
                for token in tokens
            ],
        )

        return torch.where(printable_tokens)[0]

    @cached_property
    def bad_token_ids(self) -> torch.Tensor:
        added_tokens = self.added_tokens_encoder.keys()
        tokens = [
            self.convert_tokens_to_string([token])
            for token in self.convert_ids_to_tokens(list(range(self.__vocab_size)))
        ]
        printable_tokens = torch.tensor(
            [
                token.isprintable()
                and token.isascii()
                and token not in added_tokens
                and len(token.strip()) > 0
                for token in tokens
            ],
        )
        
        bad_tokens = torch.where(~printable_tokens)[0]
        
        if self.bad_string_list == []: # no bad string list provided.
            return bad_tokens
        
        delimiters = ['.', ',', ' ', '...', '-', '_']        
        banned_strings = set(self.bad_string_list)
        
        for banned_string in self.bad_string_list:
            for delimiter in delimiters:
                banned_strings.add(delimiter.join(banned_string))
                for other_string in self.bad_string_list:
                    banned_strings.add(banned_string + delimiter + other_string)
                    banned_strings.add(other_string + delimiter + banned_string)
                
        banned_token_ids = []
        for banned_string in banned_strings:
            token_ids = self.encode(banned_string, add_special_tokens=False)
            banned_token_ids.extend(token_ids)
        
        banned_token_ids = torch.tensor(list(set(banned_token_ids)))
        combined_bad_tokens = torch.cat((bad_tokens, banned_token_ids))
        combined_bad_tokens = torch.tensor(list(set(combined_bad_tokens.tolist())))
        return combined_bad_tokens

    def pretty_decode(self, sequence: list[int], sequence_map: list[int]) -> str:
        """Decodes tokens with color highlighting based on tag mapping.

        Args:
            sequence: Token IDs to decode
            sequence_map: Mapping of tokens to tags

        Returns:
            Color-formatted decoded text

        Raises:
            ValueError: If sequence and mapping lengths don't match
        """

        if len(sequence) != len(sequence_map):
            raise ValueError(
                f"sequence_map must have same length as sequence ({len(sequence_map)} != {len(sequence)})"
            )

        # Create color map from additional special tokens where response token is green
        # and any other special token is red
        from colorlog import escape_codes

        colors = {
            token_id: escape_codes.escape_codes["bg_22"]  # dark green
            if self.convert_ids_to_tokens(token_id) == (BEGIN_TAG_FORMAT % "response")
            else escape_codes.escape_codes["bg_52"]  # dark red
            for token_id in self.additional_special_tokens_ids
        }
        colors[0] = escape_codes.escape_codes["reset"]
        colors = [colors[token_id] for token_id in sequence_map]

        # Turn tokens into colored tokens according to color map above taking care to
        # escape newlines
        tokens = self.convert_ids_to_tokens(sequence)
        tokens = [ct for color_token in zip(colors, tokens) for ct in color_token]
        decoded = self.convert_tokens_to_string(tokens)
        return decoded.replace("\n", "\\n")
