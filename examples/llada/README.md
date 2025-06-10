# Basics and requirements
Install `llmart` and download/navigate to this folder.

# Attacks on diffusion large language models (dLLMs) using `llmart`

This example shows how to use `llmart` to attack a non-autoregressive language model architecture: a masked discrete diffusion model. The victim model is the publicly available [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) model.

For more details on the principles behind masked diffusion language models, see the original [project page](https://ml-gsai.github.io/LLaDA-demo/). For users that are unfamiliar with diffusion models on continuous inputs, this [blog post](https://yang-song.net/blog/2021/score/) offers an overview from first principles.

From a hands-on researcher point of view, there is just one main difference that needs to be addressed before deploying `llmart` on this model:
```
What is the loss function?
```

In this example, we opt for using the cross-entropy loss at diffusion time `t=1`, where all response tokens are masked. This proves to be powerful enough to transfer to the entire end-to-end diffusion sampling (on a much more granular time grid). The example requires the external [generate.py](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) file from the original repo to be present in the directory, which will be automatically downloaded when running a quick test using:
```bash
make run
```

Once the file has been downloaded at least once, command-line arguments can be directly specified by running with `uv`:
```bash
uv run --with-requirements requirements.txt main.py --n_tokens 1 --suffix 20
```
