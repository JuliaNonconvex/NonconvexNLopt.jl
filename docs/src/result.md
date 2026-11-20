# Result

`optimize` returns an [`NLoptResult`](@ref) with the following fields:

- `minimizer`: the minimizer found by the algorithm
- `minimum`: the minimum objective function value found
- `problem`: the underlying `NLopt.Opt` problem, which can be inspected for
  per-algorithm statistics (see the [NLopt.jl reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/))
- `status`: a `Symbol` describing the convergence status, e.g. `:XTOL_REACHED` or `:FTOL_REACHED`
- `alg`: the `NLoptAlg` instance used
- `options`: the `NLoptOptions` instance used
- `fcalls`: the number of objective function evaluations
