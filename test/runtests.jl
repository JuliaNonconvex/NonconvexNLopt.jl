using NonconvexNLopt, LinearAlgebra, Test

@testset "Algorithm names" begin
    @test_throws ArgumentError(
        "Algorithm :AU is not a valid algorithm. The valid algorithms are (:G_MLSL, :G_MLSL_LDS, :AUGLAG, :AUGLAG_EQ, :GN_DIRECT, :GN_DIRECT_L, :GNL_DIRECT_NOSCAL, :GN_DIRECT_L_NOSCAL, :GN_DIRECT_L_RAND_NOSCAL, :GN_ORIG_DIRECT, :GN_ORIG_DIRECT_L, :GN_CRS2_LM, :GN_AGS, :GN_ESCH, :LN_COBYLA, :LN_BOBYQA, :LN_NEWUOA, :LN_NEWUOA_BOUND, :LN_PRAXIS, :LN_NELDERMEAD, :LN_SBPLX, :GD_STOGO, :GD_STOGO_RAND, :LD_CCSAQ, :LD_MMA, :LD_SLSQP, :LD_LBFGS, :LD_TNEWTON, :LD_TNEWTON_PRECOND, :LD_TNEWTON_RESTART, :LD_TNEWTON_PRECOND_RESTART, :LD_VAR1, :LD_VAR2).",
    ) NLoptAlg(:AU)

    @test_throws ArgumentError(
        "Algorithm :AUGLA is not a valid algorithm. Did you mean :AUGLAG?",
    ) NLoptAlg(:AUGLA)

    @test_throws ArgumentError(
        "A meta-algorithm :AUGLAG was input but no local optimizer was specified. Please specify a local algorithm using `NLoptAlg(:AUGLAG, local_algorithm)` where `local_optimizer` is one of the following algorithms: (:GN_DIRECT, :GN_DIRECT_L, :GNL_DIRECT_NOSCAL, :GN_DIRECT_L_NOSCAL, :GN_DIRECT_L_RAND_NOSCAL, :GN_ORIG_DIRECT, :GN_ORIG_DIRECT_L, :GN_CRS2_LM, :GN_AGS, :GN_ESCH, :LN_COBYLA, :LN_BOBYQA, :LN_NEWUOA, :LN_NEWUOA_BOUND, :LN_PRAXIS, :LN_NELDERMEAD, :LN_SBPLX, :GD_STOGO, :GD_STOGO_RAND, :LD_CCSAQ, :LD_MMA, :LD_SLSQP, :LD_LBFGS, :LD_TNEWTON, :LD_TNEWTON_PRECOND, :LD_TNEWTON_RESTART, :LD_TNEWTON_PRECOND_RESTART, :LD_VAR1, :LD_VAR2).",
    ) NLoptAlg(:AUGLAG)

    @test_throws ArgumentError(
        "Algorithm :AUGLA is not a valid algorithm. Did you mean :AUGLAG?",
    ) NLoptAlg(:AUGLA, :LD_TNEWTON_PRECON)

    @test_throws ArgumentError(
        "Algorithm :LD_TNEWTON_PRECON is not a valid algorithm. Did you mean :LD_TNEWTON_PRECOND?",
    ) NLoptAlg(:AUGLAG, :LD_TNEWTON_PRECON)
end

f(x::AbstractVector) = x[2] >= 0 ? sqrt(x[2]) : Inf
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

options = NLoptOptions(xtol_rel = 1e-4)

@testset "Simple constraints" begin
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = NLoptAlg(:LD_MMA)
    r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
    @info "Status: $(r.status)"
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end

@testset "Nested solvers" begin
    m = Model(x -> (sqrt(x[2]) + x[1]^2))
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    for alg in [
        NLoptAlg(:AUGLAG, :LD_LBFGS),
        NLoptAlg(:G_MLSL_LDS, :LD_LBFGS),
        NLoptAlg(:AUGLAG, :LD_SLSQP),
        NLoptAlg(:G_MLSL_LDS, :LD_SLSQP),
    ]
        r = NonconvexCore.optimize(m, alg, [0.1, 0.2]; options)
        @info "Status: $(r.status)"
        @test r.status == :FAILURE ||
              abs(r.minimum - 0) < 1e-6 && norm(r.minimizer .- 0) < 1e-6
    end
end

@testset "Equality constraints" begin
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

    alg = NLoptAlg(:LD_SLSQP)
    r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
    @info "Status: $(r.status)"
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end

@testset "Block constraints" begin
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))

    alg = NLoptAlg(:LD_MMA)
    r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
    @info "Status: $(r.status)"
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end

@testset "Infinite bounds" begin
    @testset "Infinite upper bound" begin
        m = Model(f)
        addvar!(m, [0.0, 0.0], [Inf, Inf])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = NLoptAlg(:LD_MMA)
        r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
        @info "Status: $(r.status)"
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
    @testset "Infinite lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [10, 10])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = NLoptAlg(:LD_MMA)
        r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
        @info "Status: $(r.status)"
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
    @testset "Infinite upper and lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [Inf, Inf])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = NLoptAlg(:LD_MMA)
        r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
        @info "Status: $(r.status)"
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
end
