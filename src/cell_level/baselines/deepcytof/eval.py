from cell_level.baselines._eval import main as _main


def main() -> None:
    _main(method="deepcytof")


if __name__ == "__main__":
    main()
