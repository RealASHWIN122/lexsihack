"""
Task Registry

Central registry for managing task specifications.
"""

from typing import Dict, List

from .specs import TaskSpec

_TASKS: Dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec, overwrite: bool = True) -> None:
    """
    Register a task specification.

    Args:
        spec: TaskSpec instance to register
        overwrite: If True (default), overwrite existing task if already registered.
                   If False, raises ValueError on duplicate names.

    Raises:
        ValueError: If task name is already registered and overwrite is False
    """
    name = spec.name.lower()
    if name in _TASKS and not overwrite:
        raise ValueError(f"Task already registered: {name}")
    _TASKS[name] = spec


def unregister_task(name: str) -> bool:
    """
    Unregister a task specification.

    Args:
        name: Task name (case-insensitive)

    Returns:
        True if the task was removed, False if it was not registered.
    """
    key = (name or "").lower()
    if key in _TASKS:
        del _TASKS[key]
        return True
    return False


def get_task(name: str) -> TaskSpec:
    """
    Get a registered task specification.

    Args:
        name: Task name (case-insensitive)

    Returns:
        TaskSpec instance

    Raises:
        ValueError: If task is not registered
    """
    key = (name or "").lower()
    if key not in _TASKS:
        raise ValueError(
            f"Unknown task '{name}'. Set the discovery config key 'task' to one "
            f"of the registered tasks: {sorted(_TASKS.keys())}."
        )
    return _TASKS[key]


def list_tasks() -> List[str]:
    """
    List all registered task names.

    Returns:
        Sorted list of registered task names
    """
    return sorted(_TASKS.keys())


def is_task_registered(name: str) -> bool:
    """
    Check if a task is registered.

    Args:
        name: Task name (case-insensitive)

    Returns:
        True if task is registered, False otherwise
    """
    return (name or "").lower() in _TASKS
