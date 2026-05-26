from mathagents.config_loader import CONFIGS_ROOT, load_solver_config, load_yaml_config

__all__ = [
    "APIClient",
    "CONFIGS_ROOT",
    "load_solver_config",
    "load_yaml_config",
]


def __getattr__(name):
    if name == "APIClient":
        from mathagents.api_client import APIClient

        return APIClient
    raise AttributeError(name)
