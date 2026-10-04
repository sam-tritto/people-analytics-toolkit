"""Deprecation warning utilities and decorators for people_analytics_toolkit."""

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    replacement: Optional[str] = None,
    deprecated_in: str = "0.1.0",
    removed_in: str = "0.3.0",
) -> Callable[[F], F]:
    """Decorator to mark functions or methods as deprecated.

    Emits a :class:`FutureWarning` on invocation informing the caller
    of the preferred alternative and scheduled removal release.

    Parameters
    ----------
    replacement : str, optional
        The recommended alternative function or method to call.
    deprecated_in : str, default="0.1.0"
        The version where deprecation was announced.
    removed_in : str, default="0.3.0"
        The future version where the feature will be removed.

    Returns
    -------
    Callable
        Decorated function emitting a FutureWarning when called.
    """
    def decorator(func: F) -> F:
        msg = (
            f"'{func.__name__}' is deprecated since v{deprecated_in} and will be removed "
            f"in v{removed_in}."
        )
        if replacement:
            msg += f" Use '{replacement}' instead."

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = True  # type: ignore
        return wrapper  # type: ignore

    return decorator


def deprecated_alias(
    target_func: Callable[..., Any],
    old_name: str,
    replacement_name: str,
    deprecated_in: str = "0.1.0",
    removed_in: str = "0.3.0",
) -> Callable[..., Any]:
    """Create a deprecated wrapper alias for a function or method.

    Emits a :class:`FutureWarning` upon invocation and appends a Sphinx-compatible
    deprecation directive to the wrapper's docstring.

    Parameters
    ----------
    target_func : Callable
        The underlying function to delegate execution to.
    old_name : str
        The legacy name being deprecated.
    replacement_name : str
        The preferred canonical function name to use.
    deprecated_in : str, default="0.1.0"
        The version where deprecation was initiated.
    removed_in : str, default="0.3.0"
        The future version where the alias will be removed.

    Returns
    -------
    Callable
        Wrapped callable emitting a warning upon execution.
    """
    msg = (
        f"'{old_name}' is deprecated since v{deprecated_in} and will be removed "
        f"in v{removed_in}. Use '{replacement_name}' instead."
    )

    @functools.wraps(target_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        return target_func(*args, **kwargs)

    wrapper.__name__ = old_name
    orig_doc = target_func.__doc__ or ""
    wrapper.__doc__ = (
        f".. deprecated:: {deprecated_in}\n"
        f"   Use :func:`{replacement_name}` instead.\n\n"
        f"{orig_doc}"
    )
    wrapper.__deprecated__ = True  # type: ignore
    return wrapper
