# Solver capabilities

[`NLoptAlg`](@ref) declares its capabilities per algorithm through
`solver_capabilities`, based on the NLopt algorithm class:

- the derivative-free local algorithms support equality constraints, inequality constraints and boxes per NLopt's own capability tables (`linear_eq`/`nonlinear_eq`, `linear_ineq`/`nonlinear_ineq`, `box` follow the selected algorithm),
- `continuous = true`; semidefinite constraints, binary variables and integer variables are **not** supported,
- `infinite_bounds` is `false` for the global algorithms that require finite bounds on every variable (e.g. `:GN_DIRECT`, `:GN_ORIG_DIRECT`),
- the meta-algorithms declare their own capabilities: `:AUGLAG`/`:AUGLAG_EQ` fold the constraints into penalties (equalities and inequalities are supported and the finite-bounds requirement is inherited from the local optimizer), while `:G_MLSL`/`:G_MLSL_LDS` support bound-constrained problems with finite bounds only.

The declaration is checked against the model via `checksubproblem!` before the solve, so unsupported model features are caught early.
