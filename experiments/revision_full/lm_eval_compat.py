"""Narrow compatibility overrides for the pinned lm-eval task definitions."""

from lm_eval.tasks import TaskManager


DATASET_PATH_OVERRIDES = {
    "truthfulqa_gen": ("truthful_qa", "truthfulqa/truthful_qa"),
}


class RevisionTaskManager(TaskManager):
    """Keep pinned task semantics while repairing renamed Hub repository IDs."""

    def _get_config(self, name):
        config = super()._get_config(name)
        override = DATASET_PATH_OVERRIDES.get(name)
        if override is None:
            return config
        old_path, namespaced_path = override
        current = config.get("dataset_path")
        if current not in {old_path, namespaced_path}:
            raise RuntimeError(
                f"Unexpected dataset_path for {name}: {current!r}; "
                f"expected {old_path!r} or {namespaced_path!r}"
            )
        config["dataset_path"] = namespaced_path
        return config
