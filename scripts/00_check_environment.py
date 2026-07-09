from __future__ import annotations

import importlib
import os
import platform
import subprocess

from _bootstrap import bootstrap

bootstrap()


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    for pkg in ["numpy", "pandas", "sklearn", "scipy", "yaml"]:
        mod = importlib.import_module(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'available')}")
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"]:
        print(f"{key}: {'set' if os.getenv(key) else 'missing'}")
    try:
        print(subprocess.check_output(["git", "--version"], text=True).strip())
    except Exception as exc:
        print(f"git unavailable: {exc}")


if __name__ == "__main__":
    main()

