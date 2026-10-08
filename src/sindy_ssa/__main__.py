"""Allow ``python -m sindy_ssa``. / 支持 ``python -m sindy_ssa``。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
