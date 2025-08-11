"""PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations"""

import importlib.metadata

__version__ = importlib.metadata.version("pdetransformer")

from . import utils
from . import visualization
from . import core
from . import data
from . import metric
from . import objectives
from . import sampler
from . import callback

__all__ = [
    "utils",
    "visualization",
    "core",
    "data",
    "metric",
    "objectives",
    "sampler",
    "callback"
]
