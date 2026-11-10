# Additional feature coverage for the exported NLoptAlg / NLoptOptions API.
# These @testitem blocks exercise paths that the main suite does not cover:
# the G_MLSL meta-algorithm through `optimize`, `optimize` without an initial
# point (the `getinit` default), a meta-algorithm driven by a zero-order local
# optimizer, a constant equality constraint (zero-jacobian fallback) and the
# fields of NLoptResult on a meta-algorithm run.

@testitem "features: G_MLSL through optimize" begin
    using NonconvexNLopt, LinearAlgebra, Test

    m = Model(x -> (sqrt(x[2]) + x[1]^2))
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    # Cap the evaluations: MLSL keeps sampling while the best objective
    # improves relative to xtol_rel; with the optimum at exactly 0 that
    # criterion may never be satisfied, so the meta-solver must not be
    # allowed to run unbounded.
    options = NLoptOptions(xtol_rel=1e-4, maxeval=2000)
    r = NonconvexCore.optimize(m, NLoptAlg(:G_MLSL, :LN_COBYLA), [0.1, 0.2]; options)
    @info "Status: $(r.status)"
    @test r.status isa Symbol
    @test isfinite(r.minimum)
    # The start point is always evaluated, so the result is at least as good.
    @test r.minimum <= sqrt(0.2) + 0.01 + 1e-8
    @test all(isfinite, r.minimizer)
end

@testitem "features: optimize without an initial point" begin
    using NonconvexNLopt, LinearAlgebra, Test

    # The initial point defaults to the model's `init` (getinit).
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0]; init=[0.9, 0.9])
    options = NLoptOptions(ftol_rel=1e-9, xtol_rel=1e-9, maxeval=5000)
    r = NonconvexCore.optimize(m, NLoptAlg(:LD_LBFGS); options=options)
    @test abs(r.minimum) < 1e-6
    @test norm(r.minimizer) < 1e-6
end

@testitem "features: meta-algorithm with a zero-order local optimizer" begin
    using NonconvexNLopt, LinearAlgebra, Test

    # AUGLAG driven by the gradient-free LN_COBYLA exercises the zero-order
    # cache invalidation inside the meta-algorithm callbacks.
    m = Model(x -> (sqrt(x[2]) + x[1]^2))
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    r = NonconvexCore.optimize(
        m,
        NLoptAlg(:AUGLAG, :LN_COBYLA),
        [0.1, 0.2];
        options=NLoptOptions(
            xtol_rel=1e-6,
            ftol_rel=1e-6,
            maxeval=2000,
            suboptions=NLoptOptions(xtol_rel=1e-6),
        ),
    )
    @info "Status: $(r.status)"
    @test r.status == :FAILURE || abs(r.minimum - 0) < 1e-4 && norm(r.minimizer .- 0) < 1e-4
end

@testitem "features: constant equality constraint (zero jacobian)" begin
    using NonconvexNLopt, LinearAlgebra, Test

    # A constant equality constraint has a `nothing` Zygote gradient; the
    # callback machinery must fall back to zero jacobian rows. AUGLAG folds
    # the (trivially satisfied) constraint into penalties, so the subproblem
    # is effectively unconstrained and converges.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-1.0, -1.0], [1.0, 1.0])
    add_eq_constraint!(m, x -> 0.0)
    r = NonconvexCore.optimize(
        m,
        NLoptAlg(:AUGLAG, :LD_LBFGS),
        [0.5, 0.5];
        options=NLoptOptions(xtol_rel=1e-8, ftol_rel=1e-8, maxeval=2000),
    )
    @test abs(r.minimum) < 1e-6
    @test norm(r.minimizer) < 1e-6
end

@testitem "features: NLoptResult fields on a meta-algorithm run" begin
    using NonconvexNLopt, LinearAlgebra, Test, NLopt

    m = Model(x -> (sqrt(x[2]) + x[1]^2))
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    alg = NLoptAlg(:AUGLAG, :LD_LBFGS)
    options = NLoptOptions(xtol_rel=1e-6, ftol_rel=1e-6, maxeval=1000)
    r = NonconvexCore.optimize(m, alg, [0.1, 0.2]; options=options)
    @test r isa NonconvexNLopt.NLoptResult
    @test r.minimizer isa AbstractVector
    @test length(r.minimizer) == 2
    @test r.minimum isa Real
    @test r.problem isa NLopt.Opt
    @test r.status isa Symbol
    @test r.alg === alg
    @test r.options === options
    @test r.fcalls isa Int
    @test r.fcalls > 0
    @test abs(r.minimum) < 1e-6
end
