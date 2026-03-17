# Modal accelsim-cutlass-trace scaffold

This directory contains the initial Modal scaffold for the `accelsim-cutlass-trace` workflow.

## Usage

Run Modal from inside this directory so the local `modal/` folder does not shadow the installed Modal package:

```sh
cd modal
python -m pip install -r requirements.txt
../.venv/bin/python -m modal run app.py::validate_environment
```

The current app only exposes a lightweight environment validation helper.
