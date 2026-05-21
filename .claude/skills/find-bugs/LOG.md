# Bug Hunting Log

## 2026-05-18

### ForwardDiff Tests - All Passed
- **Prep reuse at different points**: gradient, hessian, jacobian, value_and_gradient, derivative - all correct
- **In-place operators**: gradient!, hessian!, jacobian! - no aliasing issues, previous results unchanged
- **Second-order operators**: hvp, second_derivative prep reuse - correct
- **Consistency checks**: Hessian symmetry, pushforward/pullback duality - hold

### Empty Arrays (#802 - known issue, not filing)
- `gradient(sum, backend, Float64[])` - works
- `value_and_gradient(sum, backend, Float64[])` - **FAILS** with BoundsError
- `jacobian(identity, backend, Float64[])` - works (0x0 matrix)
- `hessian(sum, backend, Float64[])` - works (0x0 matrix)

### Operator Equivalences - All Passed
- gradient for scalar output: correct
- derivative ≈ pushforward with dx=1: holds
- jacobian columns ≈ pushforward with basis vectors: holds
- value_and_* returns exactly f(x): true
- Gradient of quadratic x'Ax = (A+A')x: holds
- Jacobian of linear Ax = A: holds

### Context Tests (#750) - All Passed (ForwardDiff)
- Single Constant context: correct
- Multiple Constant contexts: correct
- Constant with array: correct
- jacobian with Constant: correct
- Prep reuse with Constant: correct
- Changing Constant value with same prep: works correctly

### Edge Case Functions (ForwardDiff) - All Passed
- View-returning functions: works
- Tiny/huge values: works
- NaN/Inf inputs: returns [1,1,1] (expected - sum gradient)
- 3D arrays: works
- Closures with mutable state: works
- Adjoints (non-contiguous): works
- SubArrays: works
- abs at 0: returns 1.0 (ForwardDiff right-derivative, expected)
- Zero tangent pushforward: returns 0.0 (correct)

### More Edge Cases - All Passed
- Nested computations, broadcasting, reductions (sum/prod/max/min)
- norm, dot, matvec products
- Function composition, loops with conditionals
- exp(log(x)), softmax-like
- reshape, vcat, indexing operations

### Float32 / SecondOrder - All Passed
- Float32 gradient, jacobian, hessian work correctly
- Precision preserved (Float16/32/64 all maintain type)
- SecondOrder(ForwardDiff, ForwardDiff) matches plain ForwardDiff
- SecondOrder prep reuse works correctly

### In-place Functions - All Passed
- jacobian(f!, y, backend, x) works
- pushforward, pullback for in-place work
- derivative for num->vec in-place works
- Prep reuse for in-place jacobian works

### Cache Context - NOT A BUG (misunderstanding)
- Cache is for scratch space the function WRITES TO
- Function must write dual values into cache for it to work
- Read-only usage gives garbage (expected - "initial values don't matter")
- Use Constant for read-only data

### Cross-Backend: ForwardDiff vs Zygote - All Passed
- Basic gradients (sum, prod, norm, sin): match exactly
- Jacobian, Hessian: match
- Context (Constant): both handle correctly
- Prep reuse at different points: both correct
- value_and_gradient: both correct
- Challenging functions (softmax, logsumexp, cumsum, etc.): all match
- Pushforward/pullback duality: holds for both

### Combined Tests (contexts + prep reuse + edge cases) - All Passed
- Prep with Constant, call with different Constant: works
- Prep with Constant array, different array values: works
- In-place jacobian + Constant: works
- SecondOrder + Constant: works
- Multiple Constants + prep reuse: works
- Cache + prep reuse at different points: works
- HVP, pushforward, pullback with Constant: works

### Exotic Edge Cases - All Passed
- Views in intermediate computations
- Constant closures (functions as constants)
- Deeply nested computations (10x sin)
- Conditional returns
- Hessian of abs (returns zeros - correct)
- Third derivative via nesting
- Jacobian of slicing

### Batched/Special Tests - All Passed
- Chunk size variations (ForwardDiff)
- Jacobian via pushforwards matches full
- Jacobian via pullbacks matches full
- HVP matches explicit H*v
- Derivatives at boundary (t=0)

### ReverseDiff compile=true - Known Limitation Confirmed
- Basic operations work correctly
- **Control flow bug (known)**: prep at one branch, call at another → wrong result
  - This is DOCUMENTED as a limitation, not a bug to file

### Complex Numbers - BUG FILED (#1009)
- **ForwardDiff real→complex**: Returns `Complex{Dual}` instead of `ComplexF64`
  - `jacobian(x -> [complex(x[1],x[2])], AutoForwardDiff(), [1.0,2.0])` → wrong type
  - `derivative(t -> complex(t, 2t), AutoForwardDiff(), 1.0)` → wrong type
  - Filed as #1009

### Enzyme Tests - All Passed
- Basic operators: gradient, jacobian, hessian, derivative, pushforward, pullback, hvp
- Prep reuse at different points: correct
- Forward vs Reverse mode: both work, results match
- Multiple Constant contexts: works
- Empty array: works (returns `Float64[]`)
- Float32 precision: preserved
- Complex input (`abs2` of complex array): works correctly
- **Real→complex**: Works correctly (returns `ComplexF64`, unlike ForwardDiff)
- In-place jacobian with Constant: works
- Views, matrix inputs, nested functions, conditionals: all work

### Mooncake Tests - Mostly Passed
- Basic operators: gradient, jacobian, pullback, value_and_gradient - all correct
- Prep reuse at different points: correct
- Constant context: works
- Empty array: works (returns `Float64[]`)
- Float32 precision: preserved
- Complex input (`abs2` of complex array): works correctly
- Hessian/HVP: **Expected failure** - "Reverse-over-reverse not supported" (Mooncake limitation)
- **Real→complex jacobian**: Returns `Float64` matrix instead of `ComplexF64`
  - Mooncake's tangent type matches input type, ignoring output type
  - Confirmed via native API: `Mooncake.value_and_pullback!!` returns real gradients
  - This is a **Mooncake limitation**, not a DI bug
  - Scalar `derivative` works correctly (returns `ComplexF64`)
  - Filed as #1010 (for documentation/upstream)

### Real→Complex Summary
| Backend | jacobian | derivative | Native API same? |
|---------|----------|------------|------------------|
| ForwardDiff | Complex{Dual} ❌ | Complex{Dual} ❌ | Yes (both wrong) |
| Enzyme | ComplexF64 ✓ | ComplexF64 ✓ | N/A |
| Mooncake | Float64 ❌ | ComplexF64 ✓ | Yes (jacobian wrong) |

### StaticArrays Tests - All Passed
Tested with ForwardDiff and Zygote:
- **ForwardDiff**: gradient, jacobian, hessian, derivative, pushforward, pullback, hvp - all work
  - Returns SVector/SMatrix types appropriately
  - Prep reuse works correctly
  - SMatrix and MVector inputs work
  - gradient!, jacobian!, hessian! with MVector/MMatrix outputs work
- **Zygote**: gradient, jacobian, hessian, value_and_gradient, pullback - all work
  - Preserves SVector type for gradient
  - Jacobian/Hessian return Matrix{Float64} (not SMatrix) - expected behavior

### ComponentArrays Tests - Mostly Passed
- **ForwardDiff**: All operators work correctly
  - gradient, jacobian, hessian, value_and_gradient, prep reuse - all correct
  - Preserves ComponentArray structure
- **Zygote**:
  - gradient, jacobian, value_and_gradient, prep reuse - all work
  - **hessian FAILS** - "type Array has no field `a`"
    - Native `Zygote.hessian` has same issue - **Zygote limitation**, not DI bug
    - Workaround: Use SecondOrder(AutoForwardDiff(), AutoZygote()) - works correctly

### Context Tests - All Passed
- Multiple Constant contexts: ForwardDiff & Zygote both correct
- Constant with array value: ForwardDiff & Zygote both correct
- Cache context: ForwardDiff & Zygote both work (with explicit loops or broadcasting)

### Higher-Order Derivatives - All Passed
- Third/fourth derivatives via nesting: correct
- Implicit differentiation (Newton iteration): correct
- Eigenvalue/SVD derivatives: ForwardDiff fails (eigvals/svdvals don't support Dual), Zygote works

### Numerical Edge Cases - All Passed
- Very small (1e-300), very large (1e300), subnormal (5e-324) numbers: all work

### In-Place Operators
- **ForwardDiff**: jacobian, jacobian!, value_and_jacobian, pushforward, pullback, derivative for in-place - all work
- **Zygote**: All in-place operations fail - **expected** (Zygote doesn't support mutation, known limitation)

### ReverseDiff Tests - All Passed
- Basic operators: gradient, jacobian, hessian - all correct (both compile=false and compile=true)
- Prep reuse at different points: correct
- Constant context: works (single and multiple)
- In-place jacobian: works
- Empty array: works (returns `Float64[]`)
- Float32 precision: preserved
- **Control flow with compile=true**: Known limitation confirmed - prep at one branch, call at another gives wrong result

### Additional Edge Cases - All Passed
- **Broadcasting**: dimension expansion, multi-array broadcasting - all work
- **Closures with mutable state**: works correctly
- **Indexing edge cases**: slicing, reverse, step indexing - all work
- **Overlapping views**: works correctly
- **High-dimensional (n=1000)**: ForwardDiff and Zygote both work
- **Matrix-valued functions**: jacobian of matrix→matrix works
- **Nested autodiff**: differentiating through gradient works
- **Piecewise functions (abs)**: handles mixed signs correctly
- **Deep computation (10x sin)**: results match between backends
- **Type annotation (::Vector{Float64})**: ForwardDiff fails (known - requires type-generic code), Zygote works

## 2026-05-20

### Batched/Vectorized Operations - All Passed
- Sequential gradients at multiple points: works
- Prep reuse stress test (100 calls): works
- Non-square jacobians (tall, wide): work
- Scalar output/input jacobians: work
- Chunk size variations (1, 2, 5, 10): all work

### SecondOrder Combinations
- **FD/FD, FD/Zy**: hessian, hvp, second_derivative, prep reuse - all work
- **Zy/FD (reverse-over-forward)**: FAILS - ChainRulesCore projection errors
  - **Not a bug** - documentation warns "many backend combinations will fail"
  - Recommended: "forward outer over reverse inner"

### Numerical Stability - All Passed
- Discontinuities (abs, max, sign at 0): ForwardDiff handles gracefully
- Softmax, logsumexp: numerically stable
- Ill-conditioned (nearly-zero vectors): works
- Cancellation in sums: works
- Division/log near zero (1e-100, 1e-300): works

### Nothing Gradient Bug - FILED (#1011)
- Zygote returns `nothing` for zero gradients (constant functions, `sign(0.0)`, `trunc(1.5)`)
- Affects multiple operators differently:
  | Operator | Input→Output | Behavior |
  |----------|--------------|----------|
  | `derivative`, `pushforward` | scalar→scalar | MethodError |
  | `gradient`, `pullback` | array→scalar | Returns `nothing` silently |
  | `value_and_*` variants | same | Same issues |
  | `jacobian` | array→array | Works correctly (returns zeros) |
- Native Zygote returns `(nothing,)` correctly; DI doesn't convert to zeros
- **Filed as #1011** with comment about all affected operators

### Structured Matrix Types
- **Diagonal**: gradient works ✓
- **Symmetric/Hermitian**: fail - ForwardDiff limitation (native API same error)
- Not a DI bug - ForwardDiff doesn't support structured matrix seeding

### Additional Tests - All Passed
- Long computation chains (100 steps): works
- Large captured closures (10000 elements): works
- Hessian structure (diagonal, cross-terms): correct
- Reshape/permutedims operations: work

### Summary of Bugs Found (Total)
1. **#1009** - ForwardDiff real→complex returns `Complex{Dual}` instead of `ComplexF64`
2. **#1010** - Mooncake jacobian for real→complex returns `Float64` instead of `ComplexF64`
3. **#1011** - Zygote extension fails when native Zygote returns `nothing` gradient

### Known Limitations Confirmed (NOT bugs)
- Empty array: `value_and_gradient` fails with ForwardDiff (#802)
- ReverseDiff compile=true: control flow issues
- Zygote: in-place operations fail (mutation not supported)
- Zygote + ComponentArrays: hessian fails (native Zygote limitation)
- ForwardDiff: type-annotated functions fail (requires type-generic code)
- ForwardDiff: eigvals/svdvals don't support Dual numbers
- ForwardDiff: Symmetric/Hermitian matrices don't work
- SecondOrder(Zygote, ForwardDiff): reverse-over-forward fails (documented)

### Preparation System - Correctly Handles Mismatches
- Size mismatch (prep size 3, call size 5): DimensionMismatch error
- Type mismatch (prep Float64, call Float32): PreparationMismatchError
- Function mismatch (prep for f, call with g): PreparationMismatchError

### Unusual Function Patterns - All Passed
- Array wrapping/unwrapping: works
- NamedTuple field access in output: works
- Early return: works
- Try-catch blocks: works
- Nested backend calls (DI calling ForwardDiff internally): works
- @inbounds annotations: works

### Error Handling - All Correct
- Errors in user function: properly propagated
- NaN/Inf outputs: gradients computed correctly
- Non-scalar output to gradient: correctly errors
- Wrong-size output to gradient!: silent truncation (ForwardDiff behavior, not DI bug)

### Special Cases
- Constant function: ForwardDiff ✓, Zygote returns nothing (bug #1011)
- Linear function: ✓
- Identity jacobian: ✓
- Zero-Hessian (linear): ✓
- HVP with zero vector: ✓
- Large input (n=500): ✓

### FiniteDifferences Backend - All Passed (2026-05-20)
Tested `AutoFiniteDifferences(central_fdm(5, 1))`:

**Mathematical Identities:**
- Hessian symmetry: ✓
- Jacobian of linear = A: ✓
- Gradient of quadratic x'Ax = (A+A')x: ✓
- Pushforward/pullback duality: ✓
- gradient ≈ vec(jacobian) for scalar output: ✓
- HVP matches H*v: ✓

**Operator Equivalences:**
- derivative ≈ pushforward(one(x)): ✓
- Jacobian columns = pushforward with basis: ✓
- value_and_* returns correct value: ✓
- In-place matches out-of-place: N/A (no in-place for FiniteDiff)

**Prep Reuse:**
- Gradient prep reuse at different points: ✓
- Hessian prep reuse at different points: ✓
- Jacobian prep reuse at different points: ✓
- Constant context with different values: ✓

**Edge Cases:**
- View-returning functions: ✓
- reshape, vcat operations: ✓
- Deeply nested computation (20x sin): ✓
- Conditional functions: ✓
- High-dimensional (n=200): ✓
- Matrix inputs: ✓
- Matrix→Matrix Jacobian: ✓
- Rectangular Jacobians (tall, wide): ✓
- SubArray inputs: ✓
- Adjoint/Transpose inputs: ✓
- Reshaped array inputs: ✓
- Permuted array inputs: ✓

**Nested Autodiff:**
- Gradient through gradient: ✓
- Third derivative via nesting: ✓
- Implicit function (Newton sqrt): ✓

**Generic Structs (#343):**
- Mutable struct: ✓ (returns struct with gradients)
- Immutable struct: ✓
- NamedTuple: ✓
- Tuple: ✓
- **Note:** FiniteDifferences handles these naturally via its perturbation approach

**Numerical Edge Cases:**
- Very small values (1e-100): returns 0 (precision limit)
- Very large values (1e100): returns ~0 (precision limit)
- Maximum/minimum gradients: approximately correct
- Linear Hessian (should be zero): correctly zero

**Prep Size Mismatch:**
- Uses `NoGradientPrep` - doesn't actually store size info
- No error on mismatch (computed fresh each time)
- This is expected behavior for FiniteDifferences

### Tracker Tests - All Passed (2026-05-20)
- gradient, jacobian, pullback, prep reuse, Constant context: all work
- Float32 type preserved, pushforward/pullback duality holds
- Empty array works
- Expected failures: hessian (reverse-only, no nested AD), complex inputs, in-place

### PolyesterForwardDiff Tests - All Passed (2026-05-20)
- All operators (gradient, jacobian, hessian, hvp, pushforward, derivative): work
- Different chunk sizes (1, 2, 4, 8, 16): all work
- Float32, Matrix inputs, stress test (50 prep reuse iterations): work
- Results match plain ForwardDiff exactly

### FiniteDiff Tests (separate from FiniteDifferences) - 30/31 Passed
- gradient, jacobian, hessian, derivative, pushforward, prep reuse: work
- Constant, Cache, multiple contexts: work
- Empty arrays, Float32, in-place, real→complex: work
- **hvp FAILS with default settings (Bug #1012)**
  - `hvp(f, AutoFiniteDiff(), x, v)` returns 3x wrong / sign-flipped values
  - Caused by forward-over-forward FD numerical instability
  - Hessian works correctly (uses :hcentral), but HVP uses :forward default
  - Workaround: `AutoFiniteDiff(fdtype=Val(:central))`
  - **Filed as #1012**

### Same-Point Prep Variants (FiniteDiff) - All Passed
- `prepare_pushforward_same_point` then call with different tangents: ✓
- `prepare_pullback_same_point` then call with different cotangents: ✓
- `prepare_hvp_same_point` then call with different v: ✓

### Additional Edge Cases (FiniteDiff central) - All Passed
- Multi-tangent pushforward `(v1, v2, v3)`: returns tuple of correct length, all components correct
- Multi-cotangent pullback `(dy1, dy2)`: same
- Cache context with manual mutation between calls: function-overwritten cache stays correct
- `Constant` + `Cache` combined: correct
- Prep reuse with changed `Constant` value: correct (no recomputation needed)
- Matrix-valued function jacobian: correct
- Third derivative via nested `derivative` calls: correct
- Conditional branches (positive/negative): correct on both branches
- `value_and_pushforward` with matrix output: correct
- Single-element array: gradient and hessian correct

### AutoFiniteDifferences HVP Cross-Check (2026-05-20) - No Bug
Compared `hvp` vs `hessian * v` across 4 functions and 7 fdm choices
(`central_fdm/forward_fdm/backward_fdm` at orders 2, 3, 5):
- `hvp` and `hessian * v` matched bit-for-bit in every case
- High-order stencils (3, 5 points): both accurate
- Low-order stencils (2 points): both equally imprecise (forward `O(1)`,
  central `O(ε^{1/3})`) — but internally consistent
- `AutoFiniteDifferences` requires the user to pass an explicit `fdm`, so
  there is no default-precision footgun analogous to `AutoFiniteDiff`
- The #1012 inconsistency is specific to `AutoFiniteDiff`'s separate
  `fdtype` (hvp path) and `fdhtype` (hessian path) defaults
- Posted as comparison comment on #1012

### HyperHessians Tests (2026-05-20)

**Passed (no bugs):**
- `hessian`, `hvp`, `second_derivative` correctness vs ForwardDiff: matches to machine precision
- Prep reuse at different points: result at second point correct, first result not mutated
- HVP duality (`⟨u,Hv⟩ == ⟨v,Hu⟩`): exact
- `Constant`, `Cache` contexts (including reused prep with different `Constant` values): work
- `StaticArrays`: works (returns `MMatrix`)
- `value_gradient_and_hessian`, `gradient_and_hvp` (and in-place variants): work
- Chunk size variations (n=7 with chunksize 1..10): all correct
- Float32, Matrix input, heterogeneous polynomials, mixed-precision: work
- `prepare_hvp_same_point` lets you change `v` at exec time without re-prep
- Repeated calls with same prep are deterministic across chunk sweeps

**Bug filed (#1013): hessian/hvp/second_derivative fail when f does not depend on x**
- `hessian(x -> 42.0, AutoHyperHessians(), x)` → MethodError on `extract_hessian!`
- `hvp(x -> 42.0, AutoHyperHessians(), x, (v,))` → FieldError on `ϵ12`
- `second_derivative(x -> 7.0, AutoHyperHessians(), 1.5)` → FieldError on `.v`
- Also: `f(x, c) = c` with `Constant(42.0)` and `f(x) = exp(0.0)` fail the same way
- Triggers when `f(x_hyperdual)` evaluates to a plain `Float64` / `Int` without HyperDual propagation
- Native `HyperHessians.hessian` has the same failure — backend bug, DI correctly wraps
- Workaround: `f(x) = 0 * x[1] + 42.0` (forces HyperDual propagation)
- ForwardDiff handles this case, returning zeros

**Known limitation, not filed:**
- Empty array input (`Float64[]`) raises `ArgumentError: chunk size must be positive, got 0`
  (same class as #802 — empty-input inconsistency)
- `value_and_gradient(f, AutoHyperHessians(), x)`: MethodError (HH doesn't define first-order ops)
- Matrix input + matrix multiplication inside `f`: ambiguous `muladd(::HyperDual, ::HyperDual, ::HyperDual)`
  (HyperHessians.jl missing method definition)
- Strict-typed cache `c::Vector{Float64}` rejected by HyperDual cache (expected; user-side restriction)

### FastDifferentiation Tests (2026-05-20)

**Passed:**
- Basic gradient/jacobian/hessian/hvp on standard polynomials: match ForwardDiff
- Pushforward/pullback duality: holds
- Matrix `Constant` context, multiple `Constant` contexts, `Cache` context, `Constant + Cache` mixed: work
- `Constant` prep reuse with changed value (including matrices): correct
- `derivative` with vector output, scalar `Number` Constant: work
- Empty / length-1 arrays: work for `gradient`
- Matrix input via DI: gradient returns matrix-shape gradient correctly

**Known limitations (not filed):**
- Tuple / NamedTuple `Constant` (sibling of #775): unchanged — `variablize` only supports `Number` and `AbstractArray`. Maintainer ack'd in #775 as enhancement, not bug.
- Branching on `Node` value (`if x[1] > 0`): TypeError on boolean conversion — symbolic backends can't trace control flow (expected).
- Float32 input: gradient preserves type, but jacobian/hessian/derivative-of-vector return Float64 (covered by open #568).

**Bug filed (#1014): hessian returns wrong values when a variable appears inside a nonlinear op and as a multiplier**
- `hessian(v -> exp(v[1]-v[2])*v[2], AutoFastDifferentiation(), [-0.7, 2.3])`:
  H[2,2] = 0.129446, correct answer = 0.014936 (off by ~9x, error == `f(x)`)
- Affected pattern: variable `y` appears inside a nonlinear op `g(...x...y...)` AND as a separate multiplier
- Verified across operators (`exp`, `sin`, `log`, `^`) and even `(x1-x2)^2*x2` (no transcendental)
- Native `FastDifferentiation.hessian`, `sparse_hessian`, and `jacobian(jacobian([f], v), v)` all give the same wrong answer
- Manual `FastDifferentiation.derivative(FastDifferentiation.derivative([f], v[2]), v[2])` gives the correct answer
- Bug is in the Hessian/Jacobian-of-Jacobian assembly, not the derivative rules themselves
- Symbolics, HyperHessians match ForwardDiff on the same functions
- `pushforward`, `jacobian` are correct — only Hessian is wrong
- `AutoSparse(AutoFastDifferentiation())` inherits the bug
- Likely related to FastDifferentiation #65 (factorization algorithm)

### Symbolics Tests (2026-05-20)

**Passed:** gradient, hessian, hvp, jacobian, pushforward, pullback, prep reuse with different constants,
Cache context, vector context, length-1 arrays, matrix input, constant-output functions (returns `[0,0]`),
the same nonlinear*multiplier functions that break FastDifferentiation.

**Known limitations:**
- Tuple/NamedTuple `Constant`: ArgumentError (same restriction as FastDifferentiation, see #775).
- Empty array gradient: `MethodError: no method matching zero(::Type{Any})` at prep stage.
  Different failure mode from #802 but same class. Maintainer ack'd #802 as low priority.

### DifferentiateWith Tests (2026-05-20)

**Passed:**
- `DifferentiateWith(f, AutoFiniteDiff())` made differentiable via ForwardDiff, Zygote, Mooncake
- Vector-valued `f`, scalar input, scalar output, matrix input: all work
- Multiple uses of the same DW wrapper in one expression: chain rule composes correctly
- Pushforward, pullback through DW: work
- Closure over data (warned in docstring): gradient w.r.t. `x` works fine
- Mooncake DW: scalar / vector inputs and outputs all work
- Float32 inputs preserved
- Empty array input: returns empty gradient

**Expected failures (not bugs):**
- DW with `Constant`/`Cache` context: errors (docstring says contexts unsupported)
- DW around strict-typed `f(::Vector{Float64})` used in ForwardDiff Hessian or `SecondOrder(FD, X)`: errors because outer Hessian strips one Dual level, leaving inner Duals that the strict signature can't accept. Documented limitation of single-level DW.
- DW wrapping itself (nested) when the wrapped function is strict-typed: same root cause.

### GTPSA Tests (2026-05-20 cont.)

**Passed:**
- All standard operators on standard functions: gradient, jacobian, hessian, derivative,
  second_derivative, pushforward, pullback, hvp - match analytical / ForwardDiff
- Prep reuse at different points (gradient and hessian)
- Multiple `Constant` contexts, `Constant` prep reuse with changed value
- Operator equivalences (jacobian columns via pushforward; gradient ≈ vec(jacobian))
- Hessian symmetry, pushforward/pullback duality
- Empty array gradient returns `Float64[]`
- Length-1 array works
- Float32 input → Float64 gradient (type widening), values correct
- Matrix input gradient/jacobian, high-dim (n=50)
- High-order descriptors (order 10)
- Real → complex: derivative returns `ComplexF64` correctly
- FastDifferentiation #1014 nonlinear*multiplier hessian pattern: correct
- Constant + prep reuse with changed value: correct

**Known limitations (documented, not bugs):**
- `Cache` context: docs explicitly mark `AutoGTPSA` Cache as ❌
- `pullback`: docs mark `AutoGTPSA` pullback as ❌ (the fallback path returns wrong values for constant f)
- `SecondOrder(FD, GTPSA)`: `TPS{Dual}` not supported (no constructor for that combination)
- `SecondOrder(GTPSA, FD)`: extension's pushforward restricts contexts to `Vararg{Constant, C}`, breaks under SecondOrder's `FunctionContext`+`Constant` wrapping

**Bug filed (#1015): gradient/jacobian/hessian/hvp fail and pushforward returns wrong values when f does not depend on x**
- `gradient(x -> 42.0, AutoGTPSA(), [1.0, 2.0])` → MethodError (`GTPSA.gradient!(::Vector{Float64}, ::Float64)`)
- `jacobian`/`hessian`/`hvp`/`value_and_gradient`: same MethodError
- `second_derivative(t -> 42.0, AutoGTPSA(), 1.5)` → BoundsError
- `pushforward(x -> 42.0, AutoGTPSA(), x, (v,))[1]` returns `42.0` (the value) instead of `0.0`
- `pushforward(x -> [42.0, 7.0], ...)` returns `[42.0, 7.0]` instead of `[0.0, 0.0]`
- Root cause: in `onearg.jl`, when `f(xt)` evaluates to plain `Float64` instead of `TPS`,
  `yt[1]` returns the value (Julia treats Numbers as 1-elem collections) rather than the
  first-order coefficient
- Same class as #1013 (HyperHessians) but affects first-order ops and produces silently-wrong
  pushforward values
- Workaround: `f(x) = 0 * x[1] + 42.0` forces TPS propagation

### ChainRules Tests (2026-05-20)

**Passed:**
- gradient, jacobian, pullback, value_and_pullback: match ForwardDiff/Zygote on standard fns
- Pushforward via fallback works (using pullback inversion)
- Prep reuse at different points
- prepare_pullback_same_point with different cotangents
- Constant context (scalar and array)
- Matrix input, Float32 (type preserved), BigFloat
- Pullback duality holds
- Operator equivalence: gradient ≈ vec(jacobian) for vector-output cases
- Hessian via `SecondOrder(AutoForwardDiff(), AutoChainRules(...))` works
- HVP via SecondOrder works
- Empty array returns `Float64[]`
- Multiple cotangent pullback returns tuple correctly
- Constant prep reuse with changed value works

**Known limitations (not bugs):**
- `Cache` context: docs mark `AutoChainRules` Cache as ❌
- NamedTuple/Tuple input: returns `Tangent{...}` with un-unthunked inner fields (linked to #343 generic structs)
- Real → complex jacobian: returns `Matrix{Float64}` (drops imaginary part) — but this is the
  same Zygote-via-ChainRules limitation as Mooncake (#1010), not a separate bug
- Scalar-output jacobian: confusing `pick_batchsize` MethodError, but jacobian-on-scalar is
  user error anyway (ForwardDiff gives a clearer message)

**Bug filed (#1016): gradient/pullback return NoTangent instead of zeros when f does not depend on x**
- `gradient(x -> 42.0, AutoChainRules(ZygoteRuleConfig()), [1.0, 2.0])` returns
  `ChainRulesCore.NoTangent()` instead of `[0.0, 0.0]`
- `pullback`, `value_and_gradient`: same `NoTangent()` returned
- `jacobian`, `derivative`, `pushforward`: MethodError on `arroftup_to_tupofarr(::Tuple{NoTangent}, ::Float64)`
- Native `rrule_via_ad` does return `(NoTangent(), NoTangent())` (canonical zero), so the fix is in DI's `unthunk(pb(dy)[2])` path which should convert NoTangent to zeros
- Same class as #1011 (Zygote `nothing`) but different code path (ChainRulesCoreExt vs ZygoteExt)

### Mooncake Forward Tests (2026-05-20) - All Passed
Tested `AutoMooncakeForward()` and `AutoMooncakeForward(; config = Mooncake.Config(; friendly_tangents = true))`:
- gradient, jacobian, derivative, pushforward, pullback (fallback): all correct
- Prep reuse at different points
- Constant context, empty array, Float32
- Pushforward/pullback duality holds
- **Constant function**: returns zeros correctly (unlike GTPSA, HyperHessians, ChainRules)
- **Real → complex jacobian**: returns `Matrix{ComplexF64}` correctly (unlike `AutoMooncake` reverse #1010)
- SecondOrder(AutoMooncakeForward, AutoMooncake) hessian works

### Constant-Function Pattern Cross-Backend Summary
| Backend | gradient | pullback | pushforward | jacobian | hessian | derivative s→s |
|---|---|---|---|---|---|---|
| ForwardDiff | ✓ zeros | ✓ zeros | ✓ 0 | ✓ zeros | ✓ zeros | ✓ 0 |
| FiniteDiff | ✓ zeros | ✓ zeros | ✓ 0 | conf. err* | ✓ zeros | ✓ 0 |
| Zygote | ✗ nothing #1011 | ✗ nothing #1011 | MethodError | ✓ zeros | n/a | MethodError #1011 |
| ChainRules | ✗ NoTangent #1016 | ✗ NoTangent #1016 | MethodError #1016 | MethodError #1016 | n/a | MethodError #1016 |
| GTPSA | MethodError #1015 | wrong value | wrong value #1015 | MethodError #1015 | MethodError #1015 | wrong value #1015 |
| Mooncake reverse | TBD | TBD | n/a | TBD | n/a | TBD |
| Mooncake forward | ✓ zeros | ✓ zeros | ✓ 0 | ✓ zeros | n/a | ✓ 0 |
| HyperHessians | n/a | n/a | n/a | n/a | MethodError #1013 | FieldError #1013 |

*FiniteDiff scalar-output `jacobian` errors with `similar(::Float64)`. Jacobian-on-scalar is
user error across all backends, but error message clarity differs.

### Other Tests (2026-05-20) - All Passed
- BigFloat: ForwardDiff + Zygote gradient/hessian preserve type, values correct
- Float16: ForwardDiff gradient preserves type, values correct
- Rational{Int}: ForwardDiff gradient returns `Vector{Rational{Int64}}` correctly
- gradient! with Float32 buffer + Float64 input: both ForwardDiff and Zygote downcast cleanly
- gradient! with wrong-size buffer: ForwardDiff truncates silently (LOG-documented),
  Zygote raises BoundsError
- jacobian! with wrong-size matrix: DimensionMismatch (good)
- ForwardDiff chunksize > length: clear ArgumentError from ForwardDiff itself (not DI)

### Summary of Bugs Found (Total)
1. **#1009** - ForwardDiff real→complex returns `Complex{Dual}` instead of `ComplexF64`
2. **#1010** - Mooncake jacobian for real→complex returns `Float64` instead of `ComplexF64`
3. **#1011** - Zygote extension fails when native Zygote returns `nothing` gradient
4. **#1012** - FiniteDiff hvp under default `fdtype=Val(:forward)` disagrees with hessian
5. **#1013** - HyperHessians hessian/hvp/second_derivative fail when f does not depend on x
6. **#1014** - FastDifferentiation hessian wrong when variable appears in nonlinear op AND as multiplier
7. **#1015** - GTPSA gradient/jacobian/hessian/hvp fail and pushforward returns wrong values when f does not depend on x
8. **#1016** - ChainRules gradient/pullback return NoTangent instead of zeros when f does not depend on x
9. **#1017** - `SecondOrder(AutoForwardDiff(), AutoEnzyme(Reverse))` silently returns 0 from `second_derivative`
10. **#1018** - `SecondOrder(AutoEnzyme(Forward), AutoForwardDiff())` returns wrong hessian/hvp (bug in native Enzyme.Forward over ForwardDiff)

## 2026-05-20 (continued)

### Mooncake Reverse on Constant Functions - All Passed
Fills the TBD row in the constant-function cross-backend summary:
- gradient(x -> 42.0): [0.0, 0.0] ✓
- pullback: ([0.0, 0.0],) ✓
- jacobian(x -> [42.0, 7.0]): zeros ✓
- value_and_gradient, derivative, pushforward: all correct ✓
- f(x) = length(x) (no dep on x): zeros ✓
- (x, c) -> c with Constant: zeros ✓

### Enzyme on Constant Functions - Mostly Passed
- `AutoEnzyme()` (default), `AutoEnzyme(Reverse)`: all operators (gradient, pullback, pushforward, jacobian, derivative, hessian, second_derivative) return correct zeros for constant functions
- `AutoEnzyme(Forward)`: most pass, but **hessian fails** with EnzymeRuntimeException — fails for ALL functions, not just constants (forward-over-forward through `make_context_shadows` hits `jl_f__compute_sparams` which Enzyme.Forward cannot handle). Documented limitation: "many backend combinations will fail".

### ReverseDiff Exotic Types - All Passed
With both `AutoReverseDiff()` and `AutoReverseDiff(compile=true)`:
- BigFloat: gradient, hessian, jacobian all correct, preserves BigFloat
- Float32, Float16, Rational{Int}, Int: gradients correct, type preserved
- Mixed precision pushforward (BigFloat input, Float64 tangent): widens correctly
- BigFloat pullback: correct

### AutoSparse(AutoFastDifferentiation) - Inherits #1014
Confirmed #1014 buggy hessian also returned by `AutoSparse(AutoFastDifferentiation())`:
- Both dense and sparse return `H_correct[1,2]` correctly, but `H[2,2]` is wrong by ~9x for `exp(v[1]-v[2])*v[2]`
- Jacobian-of-jacobian via explicit gradient gives the correct answer
- Already noted in #1014; sparse path doesn't introduce a new failure

### Multi-tangent pushforward/pullback - Working Across Backends
- 3-tangent pushforward, 2-cotangent pullback: AutoForwardDiff, AutoZygote return correct tuples
- HVP with 2-tangent tuple: ForwardDiff, Zygote, SecondOrder(FD, Zygote) all match

### SecondOrder Combinations Filed
- **#1017**: `SecondOrder(FD, Enzyme(Reverse)) second_derivative` always 0.0 (silent)
  - Inner `derivative(t -> t^4, Enzyme.Reverse, Dual(1.5, 1.0))` returns `Dual(13.5, 0.0)` (partial silently dropped)
  - Compare: `AutoZygote` and `AutoForwardDiff` inner return `Dual(13.5, 27.0)`
  - This is the doc-recommended forward-over-reverse pattern — silent wrong is bad UX
  - Hessian/HVP with same combo fail loudly (Enzyme rejects Dual return)
- **#1018**: `SecondOrder(Enzyme(Forward), FD) hessian/hvp` returns `H_correct + gradient[i]·ones(1,n)`
  - Pattern of wrong values: each column = correct + inner_gradient
  - Reproduces in native `Enzyme.jacobian(Forward, ::, x)` applied to `x -> ForwardDiff.gradient(f, x)` — upstream Enzyme bug
  - DI correctly wraps the broken native call
- Combinations that work: SecondOrder(FD, FD), SecondOrder(FD, Zygote), SecondOrder(Enzyme(Reverse), Enzyme(Reverse)), SecondOrder(Enzyme(Forward), Enzyme(Reverse))

### Constant-Function Cross-Backend Summary (updated)
| Backend | gradient | pullback | pushforward | jacobian | hessian | derivative s→s |
|---|---|---|---|---|---|---|
| ForwardDiff | ✓ zeros | ✓ zeros | ✓ 0 | ✓ zeros | ✓ zeros | ✓ 0 |
| FiniteDiff | ✓ zeros | ✓ zeros | ✓ 0 | conf. err* | ✓ zeros | ✓ 0 |
| Zygote | ✗ nothing #1011 | ✗ nothing #1011 | MethodError | ✓ zeros | n/a | MethodError #1011 |
| ChainRules | ✗ NoTangent #1016 | ✗ NoTangent #1016 | MethodError #1016 | MethodError #1016 | n/a | MethodError #1016 |
| GTPSA | MethodError #1015 | wrong value | wrong value #1015 | MethodError #1015 | MethodError #1015 | wrong value #1015 |
| Mooncake reverse | ✓ zeros | ✓ zeros | ✓ 0 | ✓ zeros | n/a | ✓ 0 |
| Mooncake forward | ✓ zeros | ✓ zeros | ✓ 0 | ✓ zeros | n/a | ✓ 0 |
| HyperHessians | n/a | n/a | n/a | n/a | MethodError #1013 | FieldError #1013 |
| Enzyme (default/Reverse) | ✓ zeros | ✓ zeros | ✓ 0 | ✓ zeros | ✓ zeros | ✓ 0 |
| Enzyme (Forward) | ✓ zeros | ✓ zeros | ✓ 0 | ✓ zeros | EnzymeRuntimeErr** | ✓ 0 |

*FiniteDiff scalar-output `jacobian` errors with `similar(::Float64)`. Jacobian-on-scalar is
user error across all backends, but error message clarity differs.
**Enzyme(Forward) hessian fails for ALL functions, not just constants.

### Mooncake SecondOrder Combinations - All Loud
- `AutoMooncake` (reverse-only) hessian/hvp: ArgumentError "Reverse-over-reverse not supported"
- `AutoMooncake` second_derivative: ArgumentError "bitcast to differentiable type" (Mooncake explicitly guards against silently dropping tangents)
- `AutoMooncakeForward` hessian/hvp/sd: MissingIntrinsicWrapperException atomic_pointerref (Mooncake forward-over-forward limitation)
- `SecondOrder(MooncakeForward, Mooncake)` hessian/hvp: ✓ ok ; second_derivative: bitcast error
- `SecondOrder(MooncakeForward, ForwardDiff)`: all ops ✓ ok
- `SecondOrder(FD, Mooncake)`: ValueAndGradientReturnTypeError (the IEEEFloat-only check guards correctness)
- `SecondOrder(FD, MooncakeForward)`: Tangent type mismatch
- `SecondOrder(MooncakeForward, MooncakeForward)`: atomic_pointerref errors
- **Note**: Mooncake's explicit "bitcast risks dropping tangents" / "primal must be IEEEFloat" checks deliberately prevent the kind of silent-zero bug we filed for Enzyme (#1017). No new bugs.

### AutoSparse(MixedMode) Jacobian - All Passed
- `AutoSparse(MixedMode(AutoForwardDiff(), AutoMooncake()))` on bordered Jacobian (one dense row + diagonal): correct matrix
- Sanity: dense AutoForwardDiff, AutoSparse(AutoForwardDiff), AutoSparse(AutoMooncake) all match
- Tall (10×3) Jacobian matches ForwardDiff reference

### Buffer Aliasing / Unusual Buffers - All Correct
- `gradient!(f, x, backend, x)` aliasing (same array as buffer and input): gives correct result for FD and ReverseDiff
- SubArray as gradient buffer: works
- SVector as buffer: errors properly ("setindex! not defined")
- MVector input → MVector output (FD)
- Transpose buffer: works
- Wider output buffer (Float64 buf, Float32 input): widens correctly
- SubArray Hessian buffer (view(big_buf, 2:4, 2:4)): works
- SubArray input: works without mutating original

### Size Mismatch / Wrong Shapes - All Loud
- Wrong-size pushforward tangent: DimensionMismatch for FD and ReverseDiff
- Wrong-size pullback cotangent: DimensionMismatch
- `hessian` of vector-output f: DimensionMismatch with helpful "expects scalar" message (FD); shape error (ReverseDiff)
- `gradient` of vector-output f: same as above
- `gradient` of tuple-output f: same
- `derivative` on vector input: MethodError "no method matching one(::Vector)"
- `jacobian` on scalar input: DimensionMismatch with helpful "expects array" message
- Pushforward prep with wrong tangent size, then call with different size: DimensionMismatch

### Combined Operators - All Consistent
For both AutoForwardDiff and AutoReverseDiff:
- `value_gradient_and_hessian` returns matching y/g/H
- `value_and_gradient`, `gradient_and_hvp`, `value_and_jacobian`, `value_derivative_and_second_derivative`: all consistent
- `value_gradient_and_hessian!` in-place: matches out-of-place
- `value_and_gradient` matches separate `value(f, x)` + `gradient`

### AutoSparse Hessian Cross-Backend - All Passed
- Diagonal Hessian (`sum(x.^3)`): `AutoSparse(FD)`, `AutoSparse(RD)`, `AutoSparse(SecondOrder(FD,FD))`, `AutoSparse(SecondOrder(FD,RD))` all correct
- Tridiagonal Hessian: same — all correct
- Sparse hessian prep reuse at different points: correct, prior result not mutated

### Exotic Constant Types - All Passed
Tested with `AutoForwardDiff`, `AutoReverseDiff`, and `AutoMooncake`:
- `Constant{Function}` (e.g., `sin`): correct gradients
- `Constant{Symbol}` (used in control flow): correct
- `Constant{NamedTuple}`, `Constant{Tuple}`: correct
- `Constant{Int}` controlling exponent: correct
- Multiple mixed-type Constants (`Float64`, `Symbol`, `NamedTuple`) together: correct
- `Cache{Vector{Float64}}` with Mooncake: works for write-then-read pattern
- Hessian with `Constant` (scalar and NamedTuple): correct

### Linear Algebra Function Gradients - All Passed
For ForwardDiff and ReverseDiff:
- `gradient(X -> sum(X*X'), X)`: correct
- `gradient(X -> det(X), X)`: correct (matches Jacobi formula)
- `gradient(X -> tr(X*X*X), X)`: correct
- `gradient(X -> logdet(X), X)`: correct
- `gradient(x -> x'*A*x, x)`: correct (`(A+A')x`)
- `gradient(x -> sum(A\x), x)`: correct
- `gradient(x -> norm(x), x)`: correct
- Known FD limitations: `opnorm` (svdvals!), `tr(exp(X))` (exp!) error loudly — same as native FD

## 2026-05-20 (later)

### Same-Point Prep Edge Cases — All Documented Behavior, No Bugs Filed

Tested the user-suggested high-yield staleness scenarios. The docstring contract
("`other_x` must be _equal_ to `x`", "any element of `other_contexts` with type
`Constant`...must be _equal_ to the corresponding element of `contexts`") is documented
in `docs/src/explanation/operators.md`. All observed behavior conforms to that contract.

**Findings (none filed — all documented):**

1. **Same-point prep called at wrong point** (ForwardDiff, Mooncake, Zygote, ChainRules,
   Tracker, Enzyme): For backends with a default fallback (FD, Mooncake), same-point prep
   has no special caching — call at different x just works. For backends that actually
   cache (Zygote, ChainRules, Tracker), the cached `y` and `pb` closure are returned
   regardless of the passed `x` — silent stale value. Documented as user error.

2. **`value_and_pullback` + same-point prep + in-place mutation of x** (Zygote, ChainRules):
   Returns internally inconsistent `(y, tx)`:
   - `y` is a frozen snapshot from prep time → stale
   - `pb` is a closure that captures `x` by reference → refreshes when called
   - Result: `y` describes f at original `x`, `tx` describes f at mutated `x`.
   Tracker returns consistent (both stale because pb closure captures by value).
   Still documented as contract violation; not filing.

3. **Same-point prep + changing Constant** (Zygote): `check_prep` validates types but not
   values; Constants are baked into the cached closure → calling with new Constant value
   silently returns stale gradient. Conforms to documented contract.

4. **`prepare_pullback_same_point` with different cotangent**: Works correctly across
   ForwardDiff, Zygote, Enzyme (Reverse, default), Mooncake reverse/forward, ChainRules.
   Cotangent is not baked into the cache.

5. **`prepare_hvp_same_point` with different `v`**: Works correctly across ForwardDiff,
   SecondOrder(FD,Zygote), Enzyme (Reverse/default), SecondOrder(MooncakeForward,*).

### Stateful Closures (counter Ref) — Documented Limitations Confirmed
| Backend | Per-call counter increment | Gradient per call |
|---|---|---|
| ForwardDiff | ✓ | matches counter |
| Zygote | ✓ | matches counter |
| ReverseDiff(compile=false) | ✓ | matches counter |
| ReverseDiff(compile=true) | ✗ (frozen at 1) | matches frozen counter |
| Enzyme (all modes) | ✓ | matches counter |
| Mooncake (reverse) | ✗ (frozen at 0) | matches counter=1 — side-effect not propagated to user-visible Ref |
| Native Mooncake API | same as DI Mooncake | confirmed upstream |

Mooncake's not propagating closure side effects to user-visible Refs matches the native
API exactly; documented Mooncake design (tape-based, pure-function view), not a DI bug.

### Other Targeted Tests — All Passed
- **Third-order derivatives** via nested `derivative` / `second_derivative` (FD, FD/Zy
  mix): match analytic to machine precision. `AutoZygote` triple-nested errors with
  Zygote.CompileError (Zygote internal, not DI).
- **Edge dimensions**: 1×1, 1×N, M×1, 2×3 matrix-output jacobians — all correct for FD
  and Zygote.
- **Buffer aliasing**: `gradient!(f, x, backend, x)` (output buffer === input) works for
  FD and Zygote. `value_and_gradient!` with aliasing: y and g both correct (y is computed
  before g overwrites x). Aliasing g_buf and hv_buf in `gradient_and_hvp!`: last write
  wins (user error, but silent).
- **Hessian into structured matrix buffer**: `Diagonal`/`Symmetric` error with Julia's
  native restrictions on `setindex!`; not a DI issue. `SparseMatrixCSC` with insufficient
  pre-allocated entries: expands gracefully.
- **`prepare!_gradient/jacobian/hessian/hvp`** for resize 3→5: works for FD. Type-change
  blocked by `PreparationMismatchError` (documented).
- **`Cache` size**: oversized cache works (function uses only prefix); too-small cache
  errors at function level; cache contents from previous run don't affect correctness
  (function overwrites).
- **Odd patterns**: global matrices, kwargs, default args evaluated from `x`,
  view-returning functions — all correct for FD/Zygote.
- **`value_gradient_and_hessian` consistency**: matches separate `value_and_gradient` +
  `hessian` for FD, SecondOrder(FD,FD), SecondOrder(FD,Zygote), AutoSparse(FD),
  AutoSparse(SecondOrder(FD,FD)). In-place variant matches out-of-place.
- **`gradient_and_hvp` Enzyme**: AutoEnzyme(default/Reverse) match separate calls and
  analytic answer. AutoEnzyme(Forward) errors on `hvp` (forward-over-forward limitation).

### Design-Constraint Errors Confirmed (Not Bugs)
- `SecondOrder` and `AutoSparse` raise `ArgumentError: Pullback performance not defined`
  when used with `gradient`/`pullback`/`value_and_pullback`. Intentional per
  `src/utils/traits.jl:78,111`. Error message is uninformative but the design is clear.
  Not filing (enhancement, not bug).

### Summary
No new bugs filed today. Open issues remain at #1009–#1018. All same-point prep
behaviors conform to the documented contract in `docs/src/explanation/operators.md`.
The Zygote/ChainRules "stale-y, refreshed-tx" inconsistency is a subtle footgun under
contract violation, but stays within documented undefined-behavior territory.

### Backends to Skip
- **Diffractor**: unmaintained, recent releases broke the DI integration
  (see `docs/src/explanation/backends.md`). Do not include `AutoDiffractor` in any
  cross-backend test matrix; do not file bugs against it.

### To Test Next
- GPU array scenarios (if environment supports — blocked locally)

## 2026-05-21

### Type Stability (`@inferred`) Tests

**ForwardDiff:**
- Without prep: `gradient`, `jacobian`, `hessian` are type-unstable (return `Any`)
- With prep: all operators type-stable
- With explicit `chunksize`: all operators type-stable even without prep
- `derivative`, `pushforward`, `pullback`, `value_and_gradient`: type-stable without prep
- Root cause: automatic chunksize selection at runtime makes return type depend on input length
- **Not a bug** — known trade-off between convenience and type stability

**Zygote:**
- All operators (`gradient`, `jacobian`, `pullback`, `value_and_gradient`, `value_and_pullback`)
  type-stable both with and without prep

### Threading Tests (Shared Prep)

Tested concurrent calls using shared prep across 4 threads:
- Separate preps per thread: all correct (expected)
- Shared prep, different `x` values: 5% wrong results (100 iterations)
- Shared prep, same `x` value (copied): all correct
- Shared prep with `gradient!`: 12% wrong results (100 iterations)
- High contention (1000 iterations): 51% wrong results

**Conclusion:** Confirms documented behavior. Prep objects are not thread-safe; create one per thread.
Not a bug.

### DifferentiateWith + Prep Reuse Tests - All Passed

- DW wrapper with outer ForwardDiff gradient + prep reuse at different x: correct
- DW inside jacobian with prep reuse: correct
- DW hessian (forward-over-FD) + prep reuse: correct
- Nested DW (DW wrapping DW): works correctly
- DW with prep reuse + size mismatch: correctly rejected with DimensionMismatch

### HVP Edge Cases - All Passed

- Basic HVP correctness: matches H*v
- Multiple v values with same prep: all correct, previous results unchanged
- Prep reuse at different x: correct
- `prepare_hvp_same_point` with different v: works correctly
- Non-diagonal Hessian: correct
- Zero vector: returns zeros
- HVP with Constant context: correct

### Enzyme Edge Cases - All Passed

- Reverse vs Forward mode gradients agree
- Jacobian with both modes: correct
- In-place `gradient!`: correct
- Constant context: works
- Prep reuse at different points: correct
- Pushforward/pullback: correct
- Derivative (scalar): correct
- Float32: type preserved
- HVP: correct
- Empty array: works (returns `Float64[]`)
- Views (non-contiguous): works
- Forward hessian: fails as expected (EnzymeRuntimeExceptionMI) — known limitation

### Mathematical Edge Cases - All Passed

- Numerical precision near machine epsilon (1e-15 coefficients): correct
- Gradient at exact zero (minimum): returns exact zeros
- Hessian of highly curved function (`exp(sum(x.^2))`): matches analytical
- Deeply nested chain rule: correct
- Rank-deficient Jacobian: correct
- `value_gradient_and_hessian` consistency: all components match separate calls
- Pushforward with orthogonal basis vectors: correct
- HVP symmetry (`<u,Hv> = <v,Hu>`): holds
- High dimensional (n=100): correct
- Near-singular Jacobian (1e-10 entry): correct

### Mooncake Side-Effect Edge Cases - All Passed (Known Limitations Confirmed)

- Counter in closure: counter not incremented (Mooncake limitation), gradients still correct
- Global variable read (no mutation): correct
- View-returning function: correct
- Forward vs Reverse mode: agree
- Jacobian both modes: correct
- Prep reuse with side-effecting function: counter not incremented, gradients correct
- Nested closures: correct
- Prep reuse at different x: correct, previous result unchanged
- Constant context: correct
- Float32: type preserved
- Empty array: works
- Constant-output function: returns zeros correctly

**Summary:** Mooncake's side-effect masking (Ref increments not propagating) is a known
limitation documented in the native API. Gradients remain correct. No silent breakage found.

### Summary of Session
No new bugs filed. All tested areas showed expected behavior:
- Type stability trade-off in ForwardDiff is by design
- Thread safety limitations are documented
- DifferentiateWith + prep reuse works correctly
- All mathematical edge cases pass
- Mooncake side-effect masking doesn't affect gradient correctness

### Exotic Input Types (NamedTuples, Structs, Tuples) - No Bugs

**FiniteDifferences:** Handles all exotic types naturally via perturbation approach
- NamedTuple: ✓ returns NamedTuple gradient
- Tuple: ✓ returns Tuple gradient
- Mutable struct: ✓ returns struct gradient
- Immutable struct: ✓ returns struct gradient
- Nested NamedTuple: ✓

**Zygote:** Handles all exotic types
- NamedTuple, Tuple: ✓ returns same type
- Structs (mutable/immutable): ✓ returns NamedTuple (Zygote convention)

**ForwardDiff:** MethodError for all non-array types (expected - requires Dual number support)

### AutoSparse + Prep Reuse - All Passed

- Sparse Jacobian prep reuse at different x: correct, previous unchanged
- Sparse Hessian prep reuse at different x: correct, previous unchanged
- Prep at zero, call at non-zero: correct
- 20-iteration stress test: all correct
- Sparse with Constant context: correct with different c values
- SecondOrder + AutoSparse: correct

### value_and_* and In-Place Operators - All Passed

- `value_and_gradient` vs separate calls: match
- `value_and_gradient!`: correct
- `value_and_jacobian` and `value_and_jacobian!`: correct
- `value_gradient_and_hessian` and `value_gradient_and_hessian!`: correct
- In-place operators (`gradient!`, `jacobian!`, `hessian!`) with prep reuse: correct, previous unchanged
- Reusing same buffer for multiple calls: properly overwrites
- `value_and_gradient` with Constant: correct

### Combined Operators - All Passed

- `gradient_and_hvp` consistency: matches separate calls
- `gradient_and_hvp!` with tuple buffer: correct
- `value_and_derivative`: correct
- `value_derivative_and_second_derivative`: correct
- Edge dimensions (1x1, 1xN, Mx1 jacobians): all correct
- Scalar input to gradient: correctly errors (DimensionMismatch)
- Vector output to gradient: correctly errors (DimensionMismatch)
- `derivative` with vector output: correct
- `second_derivative` with vector output: correct

### ReverseDiff Edge Cases - All Passed (Known Limitations Confirmed)

- Basic gradient: `compile=false` and `compile=true` both correct
- Prep reuse with `compile=true`: correct
- Control flow with `compile=true`: wrong result when branch changes (documented limitation)
- Jacobian, Hessian: correct
- Constant context: correct
- Float32 precision: preserved
- Empty array: works
- `value_and_gradient`: correct
- In-place jacobian: correct

### Pullback/Pushforward Edge Cases - All Passed

- Pullback with multi-output function: correct (gives gradient of selected output)
- Pullback with different cotangents: matches Jacobian rows
- Pushforward with different tangents: matches Jacobian columns
- Pullback prep reuse with different cotangents: correct
- Pushforward/pullback duality: holds
- `value_and_pullback`: correct
- Pullback with zero cotangent: returns zeros
- Pullback of scalar function: matches gradient
- Pushforward with scalar input: correct
- Pushforward through view-returning function: correct

### Unusual Function Patterns - All Passed

- Reduction over broadcast (`sum(sin.(x) .* cos.(x))`): correct
- Function with `norm`: correct
- Function with `dot` product: correct
- Function with matrix-vector product: correct
- Function with `vcat`: correct
- Function with `reshape`: correct
- Function with `repeat`: correct
- Jacobian with `selectdim`: correct
- Function with `accumulate`: correct
- Function with `maximum` (non-smooth): correct subgradient
- Jacobian with `permutedims`: correct
- Very deep composition: finite values

### Cross-Backend Comparisons - All Passed

- ForwardDiff vs Zygote gradient: match
- ForwardDiff vs Zygote jacobian: match
- SecondOrder(FD, FD) vs SecondOrder(FD, Zy) hessian: match
- Complex function with multiple operations: match

### Batched Tangents - All Passed

- Batched pushforward (multiple dx): returns tuple, each matches Jacobian column
- Batched pullback (multiple dy): returns tuple, each matches Jacobian row
- Extreme values (1e-100, 1e100): correct
- Hessian symmetry for non-separable function: holds
- Prep vs no-prep results: match

### prepare!_* Resizing - Works Correctly (Documentation Clarified)

The `prepare!_*` functions return a new prep object; the `!` indicates MAY mutate (not guaranteed).
Correct usage: `prep = prepare!_gradient(f, prep, backend, new_x)` (must use return value).

- Resize smaller→larger: correct when using return value
- Resize larger→smaller: correct when using return value
- Multiple resizes: all correct
- Type change: rejected with error

### AutoSparse(MixedMode) - All Passed

`MixedMode` is designed for use inside `AutoSparse`, not standalone (documented).

- Bordered matrix pattern: correct
- Tridiagonal pattern: correct  
- Prep reuse at different x: correct, previous unchanged
- Tall jacobian (6x2): correct
- Wide jacobian (2x5): correct
- With Constant context: correct

### Closure and Multiple Context Tests - All Passed

- Large captured array (1000 elements): correct
- Mutable captured array (value at call time): correct
- Multiple Constant contexts (3 Constants): correct
- Constant array context: correct
- Constant matrix context: correct
- Nested closures: correct
- Closure factory pattern: correct
- Prep reuse with different Constant values: correct
- Hessian with Constant: correct
- Jacobian with multiple Constants: correct
- Cache context with explicit mutation: correct
- Constant + Cache combined: correct

### Tracker Backend - Mostly Passed

- Basic gradient, jacobian, pullback: all correct
- Prep reuse at different x: correct, previous unchanged
- Constant context: correct
- Float32: type preserved
- Empty array: works
- value_and_gradient: correct
- Hessian: correctly fails (reverse-only)
- Pullback duality: correct
- **Constant function gradient**: returns `nothing` instead of zeros (related to #1011)
  - `pullback` correctly returns zeros
  - `derivative` correctly returns 0.0
  - Only `gradient` returns `nothing`
  - Added comment to #1011

### Numerical Edge Cases - All Passed

- Catastrophic cancellation: very small error (~6e-9 relative)
- Very steep gradient (exp(100x)): correct
- Near-zero denominator: finite values (handles gracefully)
- Long product chain (50 elements): max relative error 2.3e-15
- Trigonometric at exact special values: exact match
- Log near 1: correct
- Very small differences (1e-14): correct
- Overflow in intermediate (log(exp(500))): correct, finite
- Underflow scenario (1e-300): correct
- Mixed scale Hessian (1e10 and 1e-10): correct

### Summary of 2026-05-21 Session

No new bugs filed. Comprehensive testing covered:
- Type stability (known trade-off in ForwardDiff)
- Threading (documented limitation)
- DifferentiateWith + prep reuse
- HVP, Enzyme, Mooncake edge cases
- Mathematical edge cases
- Exotic input types (NamedTuple, Tuple, structs)
- AutoSparse + prep reuse
- AutoSparse(MixedMode) 
- All value_and_* and in-place operators
- Combined operators (gradient_and_hvp, etc.)
- ReverseDiff edge cases (compile=true control flow documented)
- Pullback/pushforward edge cases
- Unusual function patterns (norm, dot, reshape, accumulate, maximum, vcat, repeat)
- Cross-backend comparisons (ForwardDiff vs Zygote)
- Batched tangents
- prepare!_* resizing
- Closures (large captures, mutable captures, nested, factory pattern)
- Multiple context combinations (Constant + Cache, multiple Constants)
- Numerical edge cases (catastrophic cancellation, overflow, underflow, mixed scales)
- Tracker backend (constant function gradient returns `nothing` - added comment to #1011)
- Complex numbers (confirmed #1009 behavior - ForwardDiff returns `Complex{Dual}`)
- Type edge cases: Integer, Rational, BigFloat all work correctly

All observed behaviors are either correct or documented limitations.

### To Test Next
- GPU array scenarios (if environment supports — blocked locally)

## 2026-05-21 (Session 2)

### Investigation: Type Instability in ForwardDiff Extension

Investigated comment on #1020 about type instability in DI. wsmoses noted "the code is unquestionably type unstable in DI" and provided a `code_typed` demonstration.

**Findings:**

1. **Root cause**: When using `AutoForwardDiff()` without explicit chunksize, ForwardDiff's `pickchunksize` determines chunksize at runtime. This makes `Chunk`, `JacobianConfig`, and `ForwardDiffTwoArgJacobianPrep` types have unresolved type parameters.

2. **The `dual_type` function**: `dual_type(config::JacobianConfig{T, V, N}) where {T, V, N}` pattern-matches on type parameters, requiring `Core._compute_sparams` at runtime when parameters aren't known at compile time.

3. **Performance impact** (after warmup):
   - Without explicit chunksize: ~4KB allocations, 7-10 μs
   - With explicit chunksize: ~3KB allocations, 0.4-3 μs
   - ~33% more allocations and measurably slower

4. **Distinction from #1020**: Issue #1020 was about `_compute_sparams` causing Enzyme to fail. The type instability itself is a separate (related) issue affecting performance.

5. **Workaround**: Users can specify explicit chunksize for type stability:
   ```julia
   prepare_jacobian(f!, y, AutoForwardDiff(; chunksize=N), x)
   ```

**Filed then closed:** #1021 - duplicate of #534. The type instability with automatic chunksize was already known and addressed by PR #539 (for explicit chunksize). Added comment to #1020 noting the workaround.
