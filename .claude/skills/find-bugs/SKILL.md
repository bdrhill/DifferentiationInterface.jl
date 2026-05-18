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
   - Direct backend API calls (e.g., `ForwardDiff.gradient` vs `DI.gradient`)
   - Cross-backend comparisons (e.g., ForwardDiff vs Zygote results)
   - Finite differences for ground truth
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

## Test Pattern

```julia
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using FiniteDifferences

# Reference implementation
fdm = central_fdm(5, 1)

f(x) = sum(abs2, x)
x = rand(5)

# Test gradient across backends
g_fwd = gradient(f, AutoForwardDiff(), x)
g_zyg = gradient(f, AutoZygote(), x)
g_ref = grad(fdm, f, x)[1]

@assert isapprox(g_fwd, g_ref, rtol=1e-6) "ForwardDiff gradient incorrect"
@assert isapprox(g_zyg, g_ref, rtol=1e-6) "Zygote gradient incorrect"
@assert isapprox(g_fwd, g_zyg, rtol=1e-10) "Backend mismatch"
```

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
```julia
using SparseConnectivityTracer, SparseMatrixColorings

sparse_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm()
)

f_sparse(x) = diff(x.^2)
x = rand(10)

J_sparse = jacobian(f_sparse, sparse_backend, x)
J_dense = jacobian(f_sparse, AutoForwardDiff(), x)

@assert J_sparse ≈ J_dense "Sparse jacobian incorrect"
```

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
