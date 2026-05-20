---
name: find-bugs
description: Find bugs in DifferentiationInterface by testing edge cases against reference implementations and comparing backends. Use when asked to hunt for bugs, test edge cases, or verify correctness.
allowed-tools: Bash(julia --project *) Bash(gh issue *)
---

# Find Bugs in DifferentiationInterface

Test DifferentiationInterface operators against reference implementations and cross-validate between backends.

`LOG.md` (next to this file) records what previous sessions tested, what passed, what was filed, and what's left untested. Read it before starting to avoid re-testing covered ground, and append a section at the end of the session with the date, what you tested, and what you filed.

## Workflow

1. **First**, check open issues: `gh issue list --state open`
2. Write test scripts comparing DI operators against:
   - **Cross-backend comparisons**: Same operator, different backends (e.g., `gradient(f, AutoForwardDiff(), x)` vs `gradient(f, AutoZygote(), x)`)
   - **Analytical derivatives**: For simple test functions, derive the expected result mathematically
   - **Finite differences**: Fallback for complex functions where analytical derivatives are tedious
   - **For testing correctness**: Do NOT rely on native backend API as ground truth (e.g., don't assume `ForwardDiff.gradient` is correct)
   - **For issue triage**: DO compare DI output to native API to help distinguish DI bugs from backend bugs
3. Focus on edge cases not covered by open issues
4. When a test fails or an error is raised:
   a. Skip if it matches an open issue
   b. **Check closed issues too**: `gh issue list --state closed --search "keyword"`
   c. For new bugs, minimize to an MWE and file:
      - **Always** add label `"agent"` to all issues filed
      - **Bugs** (label "bug", title "Bug:"): Incorrect results, crashes, regressions
      - **Backend issues** (title "Bug(BackendName):"): Backend-specific failures
      - **Include stacktrace**: Add collapsible `<details>` section
      - **Compare to native API**: Test the equivalent native backend call (e.g., `ForwardDiff.gradient`) to distinguish DI bugs from backend bugs
      - **Include disclaimer** at the end of every issue or comment:
        > 🤖 I am a robot. This is an experiment in agentic bug-catching under the supervision of @adrhill and @gdalle ([#1008](https://github.com/gdalle/DifferentiationInterface.jl/issues/1008)). Contents may be hallucinated.
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

These are known gaps in test coverage - start here, but also explore other areas.
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
   - **Note:** FiniteDifferences handles structs/NamedTuples/Tuples naturally; other backends may not

## Test Patterns

### 1. Prep Reuse at Different Points (Priority #1)

```julia
using DifferentiationInterface

f(x) = sum(x .^ 3)
backend = AutoForwardDiff()

x1 = [1.0, 2.0, 3.0]
x2 = [4.0, 5.0, 6.0]

# Prepare and run at x1
prep = prepare_gradient(f, backend, x1)
g1 = gradient(f, prep, backend, x1)
g1_copy = copy(g1)

# Run at different point x2
g2 = gradient(f, prep, backend, x2)

# Verify g2 is correct (analytical: 3x^2)
@assert g2 ≈ 3 .* x2 .^ 2 "Result at x2 incorrect"

# Verify g1 wasn't mutated by the second call
@assert g1 ≈ g1_copy "Previous result g1 was mutated!"

# Also test second-order operators
prep_H = prepare_hessian(f, backend, x1)
H1 = hessian(f, prep_H, backend, x1)
H1_copy = copy(H1)
H2 = hessian(f, prep_H, backend, x2)
@assert diag(H2) ≈ 6 .* x2 "Hessian at x2 incorrect"
@assert H1 ≈ H1_copy "Previous Hessian H1 was mutated!"
```

### 2. Empty and Edge-Case Arrays

```julia
backends = [AutoForwardDiff(), AutoZygote(), AutoEnzyme()]

# Empty array
x_empty = Float64[]
for b in backends
    try
        g = gradient(sum, b, x_empty)
        @assert g == Float64[] "Expected empty gradient"
    catch e
        @warn "$(typeof(b)) fails on empty array" exception=e
    end
end

# Length-1 array
x_one = [3.14]
for b in backends
    g = gradient(sum, b, x_one)
    @assert g ≈ [1.0] "Length-1 gradient incorrect for $(typeof(b))"
end
```

### 3. Context Translation Across Backends

```julia
f_ctx(x, c) = c * sum(x .^ 2)  # gradient = 2cx
x = [1.0, 2.0, 3.0]
c = 5.0

backends = [AutoForwardDiff(), AutoZygote(), AutoEnzyme()]
results = Dict()

for b in backends
    g = gradient(f_ctx, b, x, Constant(c))
    results[typeof(b)] = g
end

# All backends should agree
g_expected = 2 * c .* x
for (btype, g) in results
    @assert g ≈ g_expected "Context handling wrong for $btype"
end
```

### 4. Complex Numbers

```julia
f_complex(z) = sum(abs2, z)  # gradient = 2z (holomorphic in real/imag sense)
z = [1.0 + 2.0im, 3.0 - 1.0im]

backends = [AutoForwardDiff(), AutoZygote()]  # Not all support complex
for b in backends
    try
        g = gradient(f_complex, b, z)
        @assert g ≈ 2 .* z "Complex gradient incorrect for $(typeof(b))"
    catch e
        @warn "$(typeof(b)) fails on complex" exception=e
    end
end
```

### 5. Cross-Backend Comparison

```julia
f(x) = sum(x .^ 3)
x = rand(5)
g_analytical = 3 .* x .^ 2

backends = [AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoFiniteDifferences()]
for b in backends
    g = gradient(f, b, x)
    @assert isapprox(g, g_analytical, rtol=1e-6) "$(typeof(b)) gradient incorrect"
end
```

### 6. Using FiniteDifferences (Import Carefully)

FiniteDifferences exports `jacobian` which conflicts with DI. Use explicit imports:

```julia
using DifferentiationInterface
import DifferentiationInterface: gradient, jacobian, hessian, pushforward, pullback
import DifferentiationInterface: derivative, second_derivative, hvp
import DifferentiationInterface: prepare_gradient, prepare_jacobian, prepare_hessian
using FiniteDifferences

backend = AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1))
```

## Mathematical Identities & Consistency Checks

These can catch bugs without needing a reference implementation:

- **Hessian symmetry**: `hessian(f, backend, x)` should be symmetric for scalar-valued functions
- **Jacobian of linear function**: `jacobian(x -> A * x, backend, x) ≈ A`
- **Gradient of quadratic**: `gradient(x -> x' * A * x, backend, x) ≈ (A + A') * x`
- **Pushforward/pullback duality**: For `y = f(x)`, `dot(dy, pushforward(f, backend, x, (dx,))[1]) ≈ dot(pullback(f, backend, x, (dy,))[1], dx)` (note: tangents are tuple-wrapped)
- **Operator equivalences**:
  - `gradient(f, backend, x)` ≈ `vec(jacobian(f, backend, x))` for scalar output
  - `derivative(f, backend, x)` ≈ `pushforward(f, backend, x, (one(x),))[1]` for scalar input
  - `jacobian(f, backend, x)[:, i]` ≈ `pushforward(f, backend, x, (e_i,))[1]` where `e_i` is i-th basis vector
- **Value consistency**: `value_and_*(f, ...)` should return exactly `f(x)` as the value
- **In-place consistency**: `op!(f, result, ...)` should match `op(f, ...)`

## High-Value Test Areas

### Operators
- **First-order**: `pushforward`, `pullback`, `derivative`, `gradient`, `jacobian`
- **Second-order**: `hvp`, `hessian`, `second_derivative`
- **Variants**: `value_and_*`, `*!` (in-place), with/without preparation
- **Note:** `pushforward`, `pullback`, and `hvp` take tuple-wrapped tangents: `hvp(f, backend, x, (v,))` not `hvp(f, backend, x, v)`

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
- **Note:** Some backends use `NoGradientPrep` (e.g., FiniteDifferences) and don't actually store prep info, so size mismatch won't error

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

## Writing Style

Issue and comment text should be neutral and factual. State what happens, what should happen, show an MWE and (where useful) a table of affected cases, and stop there. The same rule applies to follow-up comments: describe behavior and cite references, not impact or sentiment.

Avoid:
- Charged adjectives: "severely wrong", "silently incorrect", "completely off", "catastrophic", "jarring", "footgun"
- Assumptions about users: "users have no clue", "user would trust the output", "would surprise users"
- Emotional intensifiers: "WAY off", "totally", "really", in all caps
- Phrases that read as complaint or advocacy rather than observation
- Emoji of any kind in issue bodies, comments, tables, or section headers. The only exception is the robot disclaimer below, which is required verbatim.

Prefer:
- Plain factual descriptions: "returns values that differ from the analytical answer by ~3x", "disagrees with hessian × v for the same backend"
- Letting MWEs, tables, and stacktraces do the persuasion
- Technical precision (cite line numbers, versions, exact error magnitudes) over rhetoric

If a sentence reads like an opinion or a complaint, rewrite it as an observation. A reader should not be able to tell whether the filer was annoyed.

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

## Native Backend Comparison
Compare DI output to the equivalent native backend API call to help distinguish DI bugs from backend bugs:

\`\`\`julia
# DI call
result_di = gradient(f, AutoForwardDiff(), x)

# Equivalent native call
result_native = ForwardDiff.gradient(f, x)

# Do they match?
result_di ≈ result_native  # true → DI correctly wraps backend (bug may be in backend)
                           # false → DI wrapper has a bug
\`\`\`

## Backend
- Backend: AutoForwardDiff() / AutoZygote() / etc.
- Works with other backends: Yes/No (list which)
- Native API gives same result: Yes/No

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

---
🤖 I am a robot. This is an experiment in agentic bug-catching under the supervision of @adrhill and @gdalle ([#1008](https://github.com/gdalle/DifferentiationInterface.jl/issues/1008)). Contents may be hallucinated.
```
