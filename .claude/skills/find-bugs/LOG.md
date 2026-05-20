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

### To Test Next
- Diffractor (forward-mode, less mature than ForwardDiff)
- GTPSA (TPSA-based backend)
- GPU array scenarios (if environment supports — blocked locally)
- ForwardDiff native (currently blocked by libquadmath.so.0 missing)
- More on `AutoSparse(...)` — see if FastDifferentiation #1014 also leaks into other sparse-backend combinations
