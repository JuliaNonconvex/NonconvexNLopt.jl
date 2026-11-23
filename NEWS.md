# NEWS

## 0.2.8

- Docs: the logo now appears only in the sidebar, not in the page body.
## 0.2.7

- Docs: package-name section titles and left-aligned logos (no code changes).
## 0.2.6

- The derivative layer (objective gradient, constraint Jacobians) now goes
  through `NonconvexCore.value_gradient` / `value_jacobian`: JuMP-converted
  models pick up the analytic JuMPModel* rrules and the objective is
  evaluated once per update. The direct Zygote dependency is dropped.

## 0.2.5

- The no-`x0` `optimize` tests run unconditionally (they were silently dead behind an always-false `isdefined` guard).
## 0.2.4

- Added docstrings for `NLoptAlg` and `NLoptOptions` (REPL help and API reference).


## 0.2.3

- Warn on the zero-gradient fallback and handle the callback keyword;
  covered the constraint callbacks at fresh points; widened the dependency
  compatibility bounds; scheduled formatter workflow.

## 0.2.1

- Widened the Zygote compat for the Core-2 ecosystem; added the
  organization logo; docs workflow fixes; covered every exported feature
  in the docs and tests.

## 0.2.0

- NonconvexCore 2 with documentation-accurate per-algorithm capability
  declarations (`infinite_bounds` per algorithm); DocumenterFragments
  docs; tests migrated to TestItems with JET/Aqua/formatting quality
  gates; Julia 1.10+.

## 0.1.13

- Documented the `optimize` name conflict with NLopt.jl in the README,
  including the recommended `import NLopt` usage for callers.

## 0.1.10

- Fixed invalid stdlib compat bounds (`LinearAlgebra`/`SparseArrays =
  "1.12.0 - 1"`) that accidentally shipped in 0.1.9 and made the package
  uninstallable on Julia 1.10/1.11.
- Requires Julia 1.10+.
