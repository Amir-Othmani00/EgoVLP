import sys

from prepare_task_steps import main


if __name__ == "__main__":
    main(["--source", "hiero", *sys.argv[1:]])
