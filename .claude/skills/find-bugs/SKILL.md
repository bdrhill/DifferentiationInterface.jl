---
name: find-bugs
description: Find bugs in DifferentiationInterface by testing edge cases against reference implementations and comparing backends. Use when asked to hunt for bugs, test edge cases, or verify correctness.
allowed-tools: Bash(julia --project *) Bash(gh issue *)
---

# Find Bugs in DifferentiationInterface

Test DifferentiationInterface operators against reference implementations and cross-validate between backends.

## Workflow

1. **First**, check open issues: `gh issue list --state open`
2. Write test scripts comparing DI operators against:
   - **Cross-backend comparisons**: Same operator, different backends (e.g., `gradient(f, AutoForwardDiff(), x)` vs `gradient(f, AutoZygote(), x)`)
   - **Analytical derivatives**: For simple test functions, derive the expected result mathematically
   - **Finite differences**: Fallback for complex functions where analytical derivatives are tedious
   - **Do NOT** compare against direct backend API calls (e.g., don't compare `DI.gradient` vs `ForwardDiff.gradient`)
3. Focus on edge cases not covered by open issues
4. When a test fails or an error is raised:
   a. Skip if it matches an open issue
   b. **Check closed issues too**: `gh issue list --state closed --search "keyword"`
   c. For new bugs, minimize to an MWE and file:
      - **Always** add label `"agent"` to all issues filed
      - **Bugs** (label "bug", title "Bug:"): Incorrect results, crashes, regressions
      - **Backend issues** (title "Bug(BackendName):"): Backend-specific failures
      - **Include stacktrace**: Add collapsible `<details>` section
   d. **Only file bugs, not feature requests**:
      - Wrong derivative values → Bug
      - Crash on supported input types → Bug
      - Performance regression → Bug (label "performance")
      - Preparation fails but operator works without prep → Bug
      - Missing support for new input type → Do not file (enhancement)
      - Missing backend support → Do not file (enhancement)
   e. **Do NOT file issues for known limitations** (see `DifferentiationInterface/docs/src/explanation/backends.md`):
      - ForwardDiff: Type restriction errors (`no method matching Float64(::Dual)`)
      - ReverseDiff: Control-flow dependent tape issues with `compile=true`
      - Zygote: Mutation errors (`Mutating arrays is not supported`)
      - Enzyme: Activity annotation complexity
      - Symbolic backends: Slow preparation
      - Thread safety: Concurrent `prep` usage (user error)
      - Complex numbers: Non-holomorphic functions
5. Continue testing other edge cases after filing

## Priority Test Areas

These are known gaps in test coverage - prioritize finding bugs here.
**Note:** These are open issues. Do NOT file duplicates. Only comment if you find significant new information not already in the issue body or comments.

1. **Prep reuse at different points** (#1007)
   - Currently DIT prepares at `zero(x)` then runs twice at same `x`
   - Test: prepare at `x1`, run at `x1`, then run at `x2`
   - Verify: result at `x2` doesn't depend on `x1`
   - Verify: after second run, values returned for `x1` weren't mutated/erased
   - Critical for all operators, especially second-order

2. **Empty/edge-case arrays** (#802)
   - `Float64[]` behaves inconsistently across backends
   - ReverseDiff, Mooncake, reverse Enzyme: return `(0.0, [])`
   - Forward Enzyme: errors (batch size 0)
   - ForwardDiff `value_and_gradient`: errors (GradientResult)
   - ForwardDiff `gradient` alone: works
   - Test all operators with empty and length-1 arrays

3. **Context translation across backends** (#750)
   - Context handling is backend-specific, may be inconsistent
   - Test `Constant`, `Cache` with same function across backends
   - Check that context values are correctly passed/translated

4. **Complex numbers** (#646)
   - Type restrictions like `M<:(AbstractMatrix{<:Real})` are too strict
   - Test `ComplexF64` inputs with various operators
   - Note: sparse differentiation excluded (SparseConnectivityTracer issue)

5. **Generic structs** (#343)
   - Custom struct inputs instead of arrays
   - Relevant for deep learning (layers, params)

## Test Pattern

```julia
using DifferentiationInterface

# Simple function with known analytical derivative
# f(x) = sum(x.^2), gradient = 2x
f(x) = sum(abs2, x)
x = rand(5)
g_analytical = 2x

# Cross-backend comparison
g_fwd = gradient(f, AutoForwardDiff(), x)
g_zyg = gradient(f, AutoZygote(), x)
g_enz = gradient(f, AutoEnzyme(), x)

@assert isapprox(g_fwd, g_analytical, rtol=1e-10) "ForwardDiff gradient incorrect"
@assert isapprox(g_fwd, g_zyg, rtol=1e-10) "ForwardDiff vs Zygote mismatch"
@assert isapprox(g_fwd, g_enz, rtol=1e-10) "ForwardDiff vs Enzyme mismatch"

# Operator consistency: gradient should match jacobian' for scalar output
J = jacobian(f, AutoForwardDiff(), x)
@assert vec(J) ≈ g_fwd "gradient vs jacobian mismatch"

# Preparation consistency
prep = prepare_gradient(f, AutoForwardDiff(), x)
g_with_prep = gradient(f, prep, AutoForwardDiff(), x)
@assert g_fwd ≈ g_with_prep "with/without prep mismatch"

# value_and_* consistency
val, g_val = value_and_gradient(f, AutoForwardDiff(), x)
@assert val ≈ f(x) "value_and_gradient returned wrong value"
@assert g_val ≈ g_fwd "value_and_gradient returned wrong gradient"
```

For complex functions, use finite differences as fallback:
```julia
g_fdm = gradient(f, AutoFiniteDifferences(), x)
@assert isapprox(g_fwd, g_fdm, rtol=1e-6) "Gradient vs finite diff mismatch"
```

## Mathematical Identities & Consistency Checks

These can catch bugs without needing a reference implementation:

- **Hessian symmetry**: `hessian(f, backend, x)` should be symmetric for scalar-valued functions
- **Jacobian of linear function**: `jacobian(x -> A * x, backend, x) ≈ A`
- **Gradient of quadratic**: `gradient(x -> x' * A * x, backend, x) ≈ (A + A') * x`
- **Pushforward/pullback duality**: For `y = f(x)`, `dot(dy, pushforward(f, backend, x, dx)) ≈ dot(pullback(f, backend, x, dy), dx)`
- **Operator equivalences**:
  - `gradient(f, backend, x)` ≈ `vec(jacobian(f, backend, x))` for scalar output
  - `derivative(f, backend, x)` ≈ `pushforward(f, backend, x, one(x))` for scalar input
  - `jacobian(f, backend, x)[:, i]` ≈ `pushforward(f, backend, x, e_i)` where `e_i` is i-th basis vector
- **Value consistency**: `value_and_*(f, ...)` should return exactly `f(x)` as the value
- **In-place consistency**: `op!(f, result, ...)` should match `op(f, ...)`

## High-Value Test Areas

### Operators
- **First-order**: `pushforward`, `pullback`, `derivative`, `gradient`, `jacobian`
- **Second-order**: `hvp`, `hessian`, `second_derivative`
- **Variants**: `value_and_*`, `*!` (in-place), with/without preparation

### Function Signatures
- Out-of-place: `f(x) = y`
- In-place: `f!(y, x) = nothing`
- With contexts: `f(x, Constant(c))`, `f(x, Cache(c))`

### Input/Output Types
- `Float64`, `Float32`, `ComplexF64`
- `Vector`, `Matrix`, `StaticArrays`
- Edge cases: length 0, length 1, very large

### Preparation Edge Cases
- Reuse with different values (same type/size)
- Reuse after type change (should error or warn)
- Same-point vs different-point preparation
- Thread safety violations (concurrent use of same prep)

### Sparse Differentiation
**Do not test sparse differentiation.** Bugs in `AutoSparse` are typically missing overloads in SparseConnectivityTracer.jl, not DifferentiationInterface bugs.

### Second-Order Operators
```julia
# Test SecondOrder combinations
backends = [
    AutoForwardDiff(),
    SecondOrder(AutoForwardDiff(), AutoZygote()),
    SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
]

f(x) = sum(x.^3)
x = rand(3)

hessians = [hessian(f, b, x) for b in backends]
for (i, H) in enumerate(hessians[2:end])
    @assert H ≈ hessians[1] "Hessian mismatch for backend $i"
end
```

### Backend-Specific Edge Cases

**ForwardDiff**: Type restrictions, custom tags, chunk sizes
```julia
# Test with explicit chunk size
backend = AutoForwardDiff(; chunksize=2)
jacobian(f, backend, rand(10))  # Should work with non-matching size
```

**Enzyme**: Activity annotations, mode selection
```julia
backend_fwd = AutoEnzyme(; mode=Enzyme.Forward)
backend_rev = AutoEnzyme(; mode=Enzyme.Reverse)
```

**ReverseDiff**: Tape compilation, control flow
```julia
# Control flow can break compiled tapes
f_branch(x) = x[1] > 0 ? sum(x) : prod(x)
backend = AutoReverseDiff(; compile=true)
# Prepare with positive x[1], test with negative
```

**Zygote**: Mutation errors
```julia
# Should fail gracefully with mutation
f_mutating!(y, x) = (y .= x.^2; nothing)
```

## Context Testing

```julia
f(x, c) = c * sum(x.^2)
x = rand(3)

# Test Constant
g1 = gradient(f, AutoForwardDiff(), x, Constant(2.0))
g2 = gradient(x -> f(x, 2.0), AutoForwardDiff(), x)
@assert g1 ≈ g2 "Constant context failed"

# Test that constant is not differentiated
# (gradient should be w.r.t. x only)
```

## Issue Format

```markdown
## Description
Brief explanation of the issue.

## MWE
\`\`\`julia
using DifferentiationInterface
using ForwardDiff: ForwardDiff  # or relevant backend

# minimal reproducing code
\`\`\`

<details>
<summary>Stacktrace</summary>

\`\`\`
ERROR: SomeError: message
Stacktrace:
 [1] function_name(args...)
   @ Module path/to/file.jl:123
\`\`\`

</details>

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Backend
- Backend: AutoForwardDiff() / AutoZygote() / etc.
- Works with other backends: Yes/No (list which)

## Environment
- Julia 1.X.Y
- DifferentiationInterface vX.Y.Z

<details>
<summary>Full environment</summary>

\`\`\`julia
julia> using Pkg; Pkg.status()
# full output

julia> using InteractiveUtils; versioninfo()
# full output
\`\`\`

</details>
```
