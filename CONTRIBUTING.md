## Code Formatting with Black

This project uses [Black](https://black.readthedocs.io/en/stable/) to automatically format Python code for consistency and readability.

### Installing Black

You can install Black using pip:

```sh
pip install black
```

### Using Black

To format all Python files in the project, run:

```sh
black .
```

You can also format a specific file:

```sh
black path/to/your_file.py
```

Please ensure your code is formatted with Black before submitting a pull request.


## Adding a Perf Model to TraceLens

If an operator is not visible in the TraceLens Perf Report, it could mean that this operator does not have a Perf Model. The respective Perf Model can be added fairly easily to TraceLens.

### 1. Mapping Operator to an exisisting Perf Model

If an operator has the exact function signature and performance model as an existing one in TraceLens, it can simply be mapped to that Perf Model from the (torch/jax)_op_mapping.py files. Here, the name of the event from the trace can be mapped directly to a Perf Model Class. 


### 2. Creating a new Perf Model

In case of a different function signature, a new Perf Model Class will have to be made for the operator.

#### Perf Model

TraceLens already defines some base classes for frequently used encountered operations (ex. GEMM, CONV, ATTN). It is recommended to overload the base classes if your operator matches any existing perf model (the most frequently added Perf Model is for variants of the Attention operator, which use the SDPA base class).

When creating a class, make sure that the details about the operator are parsed according to the function signature. Different operators store the arguments in different orders/with varying indices.

#### Op Mapping

Once the Perf Model class has been created, it needs to be mapped to the name of the event that is actually seen in the trace. Note that in some cases (ex. SDPA bwd ops) the event names must also be added in the appropriate lists so that they can be printed in the correct part of the Perf Report.