from typing import Mapping, Type, TypeVar

from .base_task import BaseObjectiveTask

T = TypeVar("T", bound=BaseObjectiveTask)

OBJECTIVE_TASK_REGISTRY: Mapping[str, Type[BaseObjectiveTask]] = {}


def register_objective_task(name: str):
    def _register(cls: Type[T]) -> Type[T]:
        OBJECTIVE_TASK_REGISTRY[name] = cls
        return cls

    return _register
