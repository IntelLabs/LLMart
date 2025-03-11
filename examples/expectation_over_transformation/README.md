# Basics and requirements
Install `llmart` and download/navigate to this folder. Run `pip install -r requirements.txt`.

# Expectation-over-Transformation (EoT) on discrete attacks using `llmart`
The example in this folder shows how to implement the EoT originally introduced in [Athalye et al., 2018](https://proceedings.mlr.press/v80/athalye18b/athalye18b.pdf) for robust adversarial attack synthesis.

The code builds on the [basic](../basic/) example and optimizes an adversarial suffix in expectation over _random token swaps_ using a Monte Carlo approach. At each optimization step, a random token position in the adversarial suffix is selected and replaced with a randomly re-sampled value, whilst also being excluded from swaps at the current step.
After a subset of the other tokens is updated, the selected token is restored to its original value, similar to how dropout operates on neural network activations.

Running the example is done with:
```bash
python main.py
```
