<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexNLopt.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexNLopt.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexNLopt.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexNLopt.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexNLopt.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexNLopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexNLopt.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexNLopt.jl

The NLopt algorithm zoo

Local and global derivative-free and gradient-based algorithms, plus meta-algorithms with pluggable local optimizers.

NonconvexNLopt.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).

## Quick start

```julia
using NonconvexNLopt

alg = NLoptAlg(:LD_SLSQP)
options = NLoptOptions()
result = optimize(model, alg, x0, options = options)
```

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexNLopt.jl/stable](https://julianonconvex.github.io/NonconvexNLopt.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexNLopt.jl/dev](https://julianonconvex.github.io/NonconvexNLopt.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexNLopt")
```
