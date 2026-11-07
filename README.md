# NonconvexNLopt

[![Build Status](https://github.com/JuliaNonconvex/NonconvexNLopt.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexNLopt.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexNLopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexNLopt.jl)

## Name conflicts

`NonconvexNLopt` reexports `optimize` from `NonconvexCore`, and
[`NLopt.jl`](https://github.com/JuliaOpt/NLopt.jl) also exports a function
named `optimize`. If both packages are loaded with `using`, the name
`optimize` becomes ambiguous (Julia prints a warning and leaves it
undefined) or resolves to the wrong package. To avoid this, load at least
one of the two packages with `import`:

```julia
import NLopt
using NonconvexNLopt

# NLopt.optimize and NonconvexNLopt.optimize stay unambiguous.
```

Alternatively, call the functions qualified: `NonconvexNLopt.optimize(...)`
and `NLopt.optimize(...)`.
