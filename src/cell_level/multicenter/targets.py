from __future__ import annotations

import argparse


_EXTERNAL_TARGETS: dict[str, tuple[str, ...]] = {
    "bln": ("bue", "vie14"),
    "bue": ("bln",),
    "vie14": ("bue",),
}


def external_test_centers(train_center: str) -> tuple[str, ...]:
    try:
        return _EXTERNAL_TARGETS[train_center]
    except KeyError as exc:
        raise ValueError(f"unknown train center: {train_center}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Print preregistered multicenter test targets")
    parser.add_argument("train_center", choices=sorted(_EXTERNAL_TARGETS))
    args = parser.parse_args()
    print(" ".join(external_test_centers(args.train_center)))


if __name__ == "__main__":
    main()
