# NEWS

## 0.1.10

- Fixed invalid stdlib compat bounds (`LinearAlgebra`/`SparseArrays =
  "1.12.0 - 1"`) that accidentally shipped in 0.1.9 and made the package
  uninstallable on Julia 1.10/1.11.
- Requires Julia 1.10+.
