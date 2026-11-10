# Various optimization algorithms from `NLopt.jl`

## Description

[NLopt](https://github.com/stevengj/nlopt) is an optimization library with a collection of optimization algorithms implemented. [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) is the Julia wrapper of `NLopt`. `NonconvexNLopt` allows the use of `NLopt.jl` using the `NLoptAlg` algorithm struct.

## Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using NLopt.
```julia
using NonconvexNLopt

alg = NLoptAlg(:LD_SLSQP)
options = NLoptOptions()
result = optimize(model, alg, x0, options = options)
```
NLopt is a dependency of NonconvexNLopt.

## Construct an instance

To construct an instance of NLopt's `NLOPT_LD_SLSQP` algorithm, use:
```julia
alg = NLoptAlg(:LD_SLSQP)
```

All the algorithms available in NLopt are:
- `:GN_DIRECT`
- `:GN_DIRECT_L`
- `:GNL_DIRECT_NOSCAL`
- `:GN_DIRECT_L_NOSCAL`
- `:GN_DIRECT_L_RAND_NOSCAL`
- `:GN_ORIG_DIRECT`
- `:GN_ORIG_DIRECT_L`
- `:GN_CRS2_LM`
- `:G_MLSL_LDS`
- `:G_MLSL`
- `:GD_STOGO`
- `:GD_STOGO_RAND`
- `:GN_AGS`
- `:GN_ISRES`
- `:GN_ESCH`
- `:LN_COBYLA`
- `:LN_BOBYQA`
- `:LN_NEWUOA`
- `:LN_NEWUOA_BOUND`
- `:LN_PRAXIS`
- `:LN_NELDERMEAD`
- `:LN_SBPLX`
- `:LD_MMA`
- `:LD_CCSAQ`
- `:LD_SLSQP`
- `:LD_LBFGS`
- `:LD_TNEWTON_PRECOND_RESTART`
- `:LD_TNEWTON_PRECOND`
- `:LD_TNEWTON_RESTART`
- `:LD_TNEWTON`
- `:LD_VAR2`
- `:LD_VAR1`
- `:AUGLAG`
- `:AUGLAG_EQ`

For a description of the above algorithms, please refer to the [algorithms section](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) of NLopt's documentation.

### Meta-algorithms and local optimizers

The meta-algorithms `:AUGLAG`, `:AUGLAG_EQ`, `:G_MLSL` and `:G_MLSL_LDS` are
not standalone algorithms: they drive a local optimizer on a sequence of
subproblems. To use one, pass the local algorithm as the second argument of
the `NLoptAlg` constructor, e.g:
```julia
alg = NLoptAlg(:AUGLAG, :LD_LBFGS)
alg = NLoptAlg(:G_MLSL_LDS, :LD_SLSQP)
```
Constructing a meta-algorithm without a local optimizer throws an
`ArgumentError`. Any of the non-meta algorithms listed above can serve as the
local optimizer.

---
**Disclaimer:**

Not all the algorithms have been tested with `Nonconvex`. So if you try one and it doesn't work, please open an issue.

---

## Options

The options keyword argument to the `optimize` function shown above must be an instance of the `NLoptOptions` struct when the algorihm is an `NLoptAlg`. To specify options use keyword arguments in the constructor of `NLoptOptions`, e.g:
```julia
options = NLoptOptions(ftol_rel = 1e-4)
```
Any option that NLopt accepts for the chosen algorithm can be passed as a
keyword argument, for example stopping tolerances and the evaluation budget:
```julia
options = NLoptOptions(xtol_rel = 1e-8, maxeval = 10_000)
```
The following tolerances have defaults of `1e-6` when not specified:

- `ftol_rel`: relative tolerance on the objective function value
- `ftol_abs`: absolute tolerance on the objective function value
- `xtol_rel`: relative tolerance on the decision variables
- `xtol_abs`: absolute tolerance on the decision variables

All the other options that can be set for each algorithm can be found in the [algorithms section](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) section of NLopt's documentation.

### Options for the local optimizer

When using a meta-algorithm, the options of the local optimizer can be set
using the `suboptions` keyword argument:
```julia
options = NLoptOptions(
    ftol_rel = 1e-4,
    suboptions = NLoptOptions(xtol_rel = 1e-8),
)
```

## Result

`optimize` returns an `NLoptResult` with the following fields:

- `minimizer`: the minimizer found by the algorithm
- `minimum`: the minimum objective function value found
- `problem`: the underlying `NLopt.Opt` problem, which can be inspected for
  per-algorithm statistics (see the [NLopt.jl reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/))
- `status`: a `Symbol` describing the convergence status, e.g. `:XTOL_REACHED` or `:FTOL_REACHED`
- `alg`: the `NLoptAlg` instance used
- `options`: the `NLoptOptions` instance used
- `fcalls`: the number of objective function evaluations

