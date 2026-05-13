"""APIRouter modules for the labelling workbench.

Each module exposes a single ``register(app, ctx)`` that attaches its
endpoints to the given FastAPI app. The shared
:class:`~src.labeling.routers._context.RouterContext` carries the
project-wide state (caches, registries, helpers) so endpoint bodies
can stay near-identical to the closure form they were extracted from.

This split was introduced because ``app.py`` had grown to 4400+ LOC
with 58 endpoints in one ``create_app()`` factory.
"""

from . import compare, exclusions, launcher, relation_reviews

__all__ = ["compare", "exclusions", "launcher", "relation_reviews"]
