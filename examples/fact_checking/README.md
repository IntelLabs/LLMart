# Basics and requirements
Install `uv` and `llmart`, and download/navigate to this folder.

## White-box attacks with `llmart`

The attacks run end-to-end adversarial optimization on the fact-checking task used by the MiniCheck framework.

MiniCheck paper: https://arxiv.org/abs/2404.10774 \
MiniCheck repository: https://github.com/Liyan06/MiniCheck

Given a claim and a document, appending adversarial suffixes for either can be run using the command below, which also downloads the required NLTK `punkt_tab` tokenizer data:
```bash
make document num_steps=1000
make claim num_steps=1000
```

> [!NOTE]
> The optimized inputs are verified with a HuggingFace re-implementation of MiniCheck's `Bespoke-MiniCheck-7B` scoring that reuses the attacked model, so no `vllm` installation or second GPU is needed.

In both cases, the goal is to adversarially make a claim unrelated to the document to be evaluated as factual and in-context for the document.

The two examples differ in how and where adversarial tokens are optimized:
- In `document.py` we use the **token replacement** functionality. In this case, all occurences of tokens that decode to `"sprawling canopy"` are automatically found and replaced with a number of `attack_budget` adversarial tokens (configurable through the command line and defaulting to 8):
https://github.com/IntelLabs/LLMart/blob/1f954133b935b6829db3fb5c0d7f84b9d94f2490/examples/fact_checking/document.py#L58-L63
- In `claim.py` we use the **adversarial suffix** functionality, which appends additional tokens at the end of the input claim:
https://github.com/IntelLabs/LLMart/blob/1f954133b935b6829db3fb5c0d7f84b9d94f2490/examples/fact_checking/claim.py#L51
