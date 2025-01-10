# Basics and requirements
Install `llmart`, download this folder, and run `pip install -r requirements.txt`.

## White-box attacks with `llmart`

The attacks run end-to-end adversarial optimization on the fact-checking task used by the MiniCheck framework.

MiniCheck paper: https://arxiv.org/abs/2404.10774 \
MiniCheck repository: https://github.com/Liyan06/MiniCheck

Given a claim and a document, appending adversarial suffixes for either can be run using the commands:
```
CUDA_VISIBLE_DEVICES=0,1 python document.py
CUDA_VISIBLE_DEVICES=0,1 python claim.py
```

In both cases, the goal is to adversarially make a claim unrelated to the document to be evaluated as factual and in-context for the document.

The two examples differ in how and where adversarial tokens are optimized:
- In `document.py` we use the **token replacement** functionality. In this case, all occurences of tokens that decode to `"sprawling canopy"` are automatically found and replaced with a number of `attack_budget` adversarial tokens (configurable through the command line and defaulting to 8):
https://github.com/intel-sandbox/llmart/blob/4db2ac24c50d5cefa1f0e2aa40c168872c93467e/examples/fact_checking/document.py#L58-L63
- In `claim.py` we use the **adversarial suffix** functionality, which appends additional tokens at the end of the input claim:
https://github.com/intel-sandbox/llmart/blob/4db2ac24c50d5cefa1f0e2aa40c168872c93467e/examples/fact_checking/claim.py#L51
