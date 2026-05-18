# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DifferentiationInterface.jl provides a unified API for automatic differentiation (AD) in Julia, supporting 15+ backends (ForwardDiff, Enzyme, Zygote, etc.) through a common interface.

**Monorepo structure:**
- `DifferentiationInterface/` - Core package with unified AD API
- `DifferentiationInterfaceTest/` - Testing and benchmarking utilities

## Common Commands

### Running Tests

**Core tests (no external AD backends):**
```bash
cd DifferentiationInterface && julia --project -e 'using Pkg; Pkg.test()'
```

**Single backend test** (e.g., ForwardDiff):
```bash
JULIA_DI_TEST_GROUP=ForwardDiff julia --project=DifferentiationInterface/test/Back/ForwardDiff -e '
using Pkg
Pkg.develop([PackageSpec(path="DifferentiationInterface"), PackageSpec(path="DifferentiationInterfaceTest")])
include("DifferentiationInterface/test/Back/run_backend.jl")'
```

**DifferentiationInterfaceTest tests:**
```bash
cd DifferentiationInterfaceTest && julia --project -e 'using Pkg; Pkg.develop(path="../DifferentiationInterface"); Pkg.test()'
```

### Code Formatting

Uses [Runic.jl](https://github.com/fredrikekre/Runic.jl) via pre-commit:
```bash
pre-commit run --all-files
```

## API Reference

### Operators

**8 operators**, each with 4 variants (`op`, `op!`, `value_and_op`, `value_and_op!`):

| Operator | Order | Input | Output | Result |
|----------|-------|-------|--------|--------|
| `pushforward` | 1 | Any | Any | JVP (same shape as y) |
| `pullback` | 1 | Any | Any | VJP (same shape as x) |
| `derivative` | 1 | Number | Any | dy/dx (same shape as y) |
| `gradient` | 1 | Array | Number | grad (same shape as x) |
| `jacobian` | 1 | Array | Array | matrix (length(y) x length(x)) |
| `hvp` | 2 | Array | Number | Hessian-vector product |
| `hessian` | 2 | Array | Number | matrix (length(x) x length(x)) |
| `second_derivative` | 2 | Number | Any | d2y/dx2 |

### Function Forms

```julia
y = f(x, contexts...)   # out-of-place
f!(y, x, contexts...)   # in-place (only pushforward, pullback, derivative, jacobian)
```

**Mutation rule**: Either an argument's value matters OR it can be mutated, never both.
- `x` cannot be mutated (differentiation point)
- `y` in in-place functions must be entirely overwritten

### Preparation

```julia
prep = prepare_gradient(f, backend, x)      # prepare once
gradient(f, prep, backend, x)               # fast repeated calls
gradient!(f, grad, prep, backend, x)        # in-place variant
```

- `prep` is **mutated** during calls and **NOT thread-safe**
- Reuse requires same types and sizes (values can differ)
- `prepare_*_same_point` variants exist when input won't change

### Contexts

Additional arguments that are not differentiated:

```julia
gradient(f, backend, x, Constant(c))  # c influences output, cannot be mutated
gradient(f, backend, x, Cache(c))     # c is scratch space, can be mutated
```

- `Constant` - value matters, no mutation allowed
- `Cache` - initial value arbitrary, mutation allowed
- `ConstantOrCache` - hybrid with both constant and cache parts

### Sparse Differentiation

Only for `jacobian` and `hessian`:

```julia
sparse_backend = AutoSparse(
    backend;
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm()
)
```

**Mixed mode** for Jacobians with dense rows AND columns:
```julia
AutoSparse(MixedMode(AutoForwardDiff(), AutoZygote()); ...)
```

### Second Order

```julia
SecondOrder(outer_backend, inner_backend)  # forward-over-reverse typically best
```

## Architecture

### Source Structure (`DifferentiationInterface/src/`)

- `first_order/` - pushforward, pullback, derivative, gradient, jacobian
- `second_order/` - hvp, hessian, second_derivative
- `utils/` - context, prep, traits, basis, batchsize, check, errors
- `misc/` - differentiate_with, from_primitive, sparsity_detector, zero_backends

### Backend Extensions (`DifferentiationInterface/ext/`)

Each extension implements:
1. `DI.check_available(::AutoBackend) = true`
2. Primitive operations (`pushforward` for forward-mode, `pullback` for reverse-mode)
3. Optionally: custom high-level operators for efficiency

Extension naming: `DifferentiationInterface<Backend>Ext/`

### Testing Infrastructure

`DifferentiationInterfaceTest` provides:
- `test_differentiation(backend, scenarios)` - main entry point
- `default_scenarios()`, `sparse_scenarios()`, `static_scenarios()`, `component_scenarios()`
- Type stability and correctness checks

Test groups via environment variables:
- `JULIA_DI_TEST_GROUP` - Core test group or backend name
- `JULIA_DIT_TEST_GROUP` - DifferentiationInterfaceTest group

## Important Limitations

1. **Thread safety**: `prep` objects are not thread-safe; create one per thread
2. **Single active argument**: Only `x` is differentiated; use contexts for other args
3. **Complex numbers**: Only holomorphic functions supported (experimental)
4. **Sparse prep reuse**: Cannot reuse if sparsity pattern changes

## Backend-Specific Notes

| Backend | Key Consideration |
|---------|-------------------|
| ForwardDiff | Needs type-generic code; use `eltype(x)` not `Float64` |
| ReverseDiff | `compile=true` records control flow; wrong branches = wrong results |
| Zygote | No mutation support; use ChainRulesCore `rrule` for workarounds |
| Enzyme | Most flexible but activity handling tricky via DI; consider native API |
| Symbolic (FastDifferentiation, Symbolics) | Preparation very slow; built-in sparsity |

## Conventions

- [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
- Julia 1.10+ required
- Backend types from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)
