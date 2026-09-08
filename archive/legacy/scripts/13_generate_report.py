from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

from cents.reporting import generate_report


if __name__ == "__main__":
    print(generate_report())

