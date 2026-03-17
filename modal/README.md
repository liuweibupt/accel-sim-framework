# Modal accelsim-cutlass-trace scaffold

This package defines the `accelsim-cutlass-trace` Modal application. It currently exposes a lightweight environment validation helper.

## Usage

Install the Modal dependency and run the helper via the python module mode:

```sh
python -m pip install -r modal/requirements.txt
python -m modal modal.app validate_environment
```

Using the `python -m modal ...` entry point keeps the tooling aligned with package-style workflows while this scaffold remains minimal.
