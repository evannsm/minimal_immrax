# minimal_immrax without NN functionality of linrax to avoid issues I was having with installing and hopefully make RPi install easier

`immrax` is a tool for interval analysis and mixed monotone reachability analysis in JAX.

Inclusion function transformations are composable with existing JAX transformations, allowing the use of Automatic Differentiation to learn relationships between inputs and outputs, as well as parallelization and GPU capabilities for quick, accurate reachable set estimation.

For more information, please see the full [documentation](https://immrax.readthedocs.io).

## Installation

### Setting up a `conda` environment

We recommend installing JAX and `immrax` into a `conda` environment ([miniconda](https://docs.conda.io/projects/miniconda/en/latest/)).

```shell
conda create -n immrax python=3.11
conda activate immrax
```

### Installing immrax

For now, manually clone the Github repository and `pip install` it. We plan to release a stable version on PyPi soon.

```shell
git clone https://github.com/gtfactslab/immrax.git
cd immrax
pip install .
```
