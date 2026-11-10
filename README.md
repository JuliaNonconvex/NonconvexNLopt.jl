<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexNLopt.jl logo">
</p>

# NonconvexNLopt


[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexNLopt.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexNLopt.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexNLopt.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexNLopt.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexNLopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexNLopt.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

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
