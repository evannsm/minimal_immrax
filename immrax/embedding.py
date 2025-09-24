import abc
from functools import partial
from typing import Any, Callable, Literal, Union

import jax
import jax.numpy as jnp
from jaxtyping import Float, Integer, Bool, Array

from .inclusion import Interval, i2ut, interval, jacif, mjacif, natif, ut2i
from .system import LiftedSystem, System

__all__ = [
    "EmbeddingSystem",
    "InclusionEmbedding",
    "TransformEmbedding",
    "ifemb",
    "natemb",
    "jacemb",
    "mjacemb",
    "embed",
    "get_faces",
]


class EmbeddingSystem(System, abc.ABC):
    """EmbeddingSystem

    Embeds a System

    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n`

    into an Embedding System evolving on the upper triangle

    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n}.
    """

    sys: System

    @abc.abstractmethod
    def E(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        """The right hand side of the embedding system.

        Parameters
        ----------
        t : Union[Integer, Float]
            The time of the embedding system.
        x : jax.Array
            The state of the embedding system.
        *args :
            interval-valued control inputs, disturbance inputs, etc. Depends on parent class.
        **kwargs :


        Returns
        -------
        jax.Array
            The time evolution of the state on the upper triangle

        """

    def f(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.E(t, x, *args, **kwargs)


class InclusionEmbedding(EmbeddingSystem):
    """EmbeddingSystem

    Embeds a System

    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n`,

    into an Embedding System evolving on the upper triangle

    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n},

    using an Inclusion Function for the dynamics f.
    """

    sys: System
    F: Callable[..., Interval]
    Fi: Callable[..., Interval]

    def __init__(
        self,
        sys: System,
        F: Callable[..., Interval],
        Fi: Callable[..., Interval] | None = None,
    ) -> None:
        """Initialize an EmbeddingSystem using a System and an inclusion function for f.

        Args:
            sys (System): The system to be embedded
            if_transform (InclusionFunction): An inclusion function for f.
        """
        self.sys = sys
        self.F = F
        self.evolution = sys.evolution
        self.xlen = sys.xlen * 2

    def E(
        self,
        t: Any,
        x: jax.Array,
        *args,
        refine: Callable[[Interval], Interval] | None = None,
        **kwargs,
    ) -> jax.Array:
        t = interval(t)
        # jax.debug.print("isnan: {0}", jnp.isnan(x).any())

        if refine is not None:
            convert = lambda x: refine(ut2i(x))
            Fkwargs = lambda t, x, *args: self.F(t, refine(x), *args, **kwargs)
        else:
            convert = ut2i
            Fkwargs = partial(self.F, **kwargs)

        x_int = convert(x)
        # jax.debug.print(
        #     "lower: {0}, upper: {1}",
        #     jnp.isnan(x_int.lower).any(),
        #     jnp.isnan(x_int.upper).any(),
        # )

        if self.evolution == "continuous":
            n = self.sys.xlen
            _x = x_int.lower
            x_ = x_int.upper

            # Computing F on the faces of the hyperrectangle

            _X = interval(
                jnp.tile(_x, (n, 1)), jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1)))
            )
            _E = jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, _X, *args)

            X_ = interval(
                jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))), jnp.tile(x_, (n, 1))
            )
            E_ = jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, X_, *args)

            # return jnp.concatenate((_E, E_))
            output = jnp.concatenate((jnp.diag(_E.lower), jnp.diag(E_.upper)))
            # jax.debug.print("output isnan: {0}", jnp.isnan(output).any())
            return jnp.concatenate((jnp.diag(_E.lower), jnp.diag(E_.upper)))

        elif self.evolution == "discrete":
            # Convert x from ut to i, compute through F, convert back to ut.
            return i2ut(self.F(interval(t), x_int, *args, **kwargs))
        else:
            raise Exception("evolution needs to be 'continuous' or 'discrete'")


def ifemb(sys: System, F: Callable[..., Interval]):
    """Creates an EmbeddingSystem using an inclusion function for the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed
    F : Callable[..., Interval]
        Inclusion function for the dynamics of sys.

    Returns
    -------
    EmbeddingSystem
        Embedding system from the inclusion function transform.

    """
    return InclusionEmbedding(sys, F)


def embed(F: Callable[..., Interval]):
    def E(
        t: Any,
        x: jax.Array,
        *args,
        refine: Callable[[Interval], Interval] | None = None,
        **kwargs,
    ):
        n = len(x) // 2
        _x = x[:n]
        x_ = x[n:]

        if refine is not None:
            Fkwargs = lambda t, x, *args: F(t, refine(x), *args, **kwargs)
        else:
            Fkwargs = partial(F, **kwargs)

        # Computing F on the faces of the hyperrectangle

        if n > 1:
            _X = interval(
                jnp.tile(_x, (n, 1)), jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1)))
            )
            _E = interval(
                jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, _X, *args)
            )

            X_ = interval(
                jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))), jnp.tile(x_, (n, 1))
            )
            E_ = interval(
                jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, X_, *args)
            )
            return jnp.concatenate((jnp.diag(_E.lower), jnp.diag(E_.upper)))
        else:
            _E = Fkwargs(t, interval(_x)).lower
            E_ = Fkwargs(t, interval(x_)).upper
            return jnp.array([_E, E_])

    return E


def get_faces(ix: Interval) -> tuple[Interval, Interval]:
    n = len(ix)

    _x = ix.lower
    x_ = ix.upper

    # _X = interval(
    #     , jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1)))
    # )

    # X_ = interval(
    #     , jnp.tile(x_, (n, 1))
    # )

    X = interval(
        jnp.vstack(
            (jnp.tile(_x, (n, 1)), jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))))
        ),
        jnp.vstack(
            (jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1))), jnp.tile(x_, (n, 1)))
        ),
    )

    return X


class TransformEmbedding(InclusionEmbedding):
    def __init__(self, sys: System, if_transform=natif) -> None:
        """Initialize an EmbeddingSystem using a System and an inclusion function transform.

        Parameters
        ----------
        sys : System
            _description_
        if_transform : IFTransform
            _description_. Defaults to natif.

        Returns
        -------

        """
        F = if_transform(sys.f)
        # Fi = [if_transform(sys.fi[i]) for i in range(sys.xlen)]
        super().__init__(sys, F)


def natemb(sys: System):
    """Creates an EmbeddingSystem using the natural inclusion function of the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed

    Returns
    -------
    EmbeddingSystem
        Embedding system from the natural inclusion function transform.

    """
    return TransformEmbedding(sys, if_transform=natif)


def jacemb(sys: System):
    """Creates an EmbeddingSystem using the Jacobian-based inclusion function of the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed

    Returns
    -------
    EmbeddingSystem
        Embedding system from the Jacobian-based inclusion function transform.

    """
    return TransformEmbedding(sys, if_transform=jacif)


def mjacemb(sys: System):
    """Creates an EmbeddingSystem using the Mixed Jacobian-based inclusion function of the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed

    Returns
    -------
    EmbeddingSystem
        Embedding system from the Mixed Jacobian-based inclusion function transform.

    """
    return TransformEmbedding(sys, if_transform=mjacif)

