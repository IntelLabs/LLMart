# Basics and requirements
Install `llmart` and download/navigate to this folder.

## Forcing random string responses
This application uses `llmart` to token-force pseudo-random string responses to basic questions. You can first generate a random alphanumeric (+normal special characters) of length *10* using:
```bash
tr -dc 'A-Za-z0-9!#%&'\''()*+,-./:;<=>?@[\]^_{|}~' </dev/urandom | head -c 10; echo
```

Using the output, we can optimize an adversarial suffix that forces this response via:
```bash
uv run python whitebox.py "O4c9I=9*ih"
```

Any string passed as argument will be the target. Longer strings will typically require significantly more optimization steps:
```bash
uv run python whitebox.py "r*7It0Z>Mg-qoYip-]Ji;|'bcq;}b;Mm^'|Cq0+0U!!dfO6q(p}el*N8Nvu(T4xK<m;k{?qab(OgCP:0;Uk_@/q}se#cYx.P1" --max_steps=20000
```

See `more_strings.txt` for more pre-generated random strings. You can customize other hyper-parameters via CLI arguments:
```bash
usage: whitebox.py [-h] [--max_steps MAX_STEPS] [--num_tokens MAX_OPTIM_TOKENS] [--lr LR] [--use_hard_tokens] sequence

positional arguments:
  sequence              Target generation/completion.

options:
  -h, --help            show this help message and exit
  --max_steps MAX_STEPS
                        Optimise prompt for no more than N steps.
  --num_tokens MAX_OPTIM_TOKENS
                        Number of optimized tokens
  --lr LR               Learning rate for adversarial optimization.
  --use_hard_tokens     Find hard tokens instead of soft tokens in the emebdding space.
```
