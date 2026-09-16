"""Hidden helper code of the MP5 notebooks (LLM vision lab). The notebooks only call `lab`."""
try:
    from .lab import lab  # noqa: F401
except ImportError:      # build scripts import config before the lab module exists / without its dependencies
    pass
