using NonconvexNLopt, LinearAlgebra, Test

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a*x[1] + b)^3 - x[2]

options = NLoptOptions(xtol_rel = 1e-4)

@testset "Simple constraints" begin
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = NLoptAlg(:LD_MMA)
    r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
    @info "Status: $(r.status)"
    @test abs(r.minimum - sqrt(8/27)) < 1e-6
    @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
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
        @test r.status == :FAILURE || abs(r.minimum - 0) < 1e-6  && norm(r.minimizer .- 0) < 1e-6
    end
end

@testset "Equality constraints" begin
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    add_eq_constraint!(m, x -> sum(x) - 1/3 - 8/27)

    alg = NLoptAlg(:LD_SLSQP)
    r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
    @info "Status: $(r.status)"
    @test abs(r.minimum - sqrt(8/27)) < 1e-6
    @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
end

@testset "Block constraints" begin
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))

    alg = NLoptAlg(:LD_MMA)
    r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
    @info "Status: $(r.status)"
    @test abs(r.minimum - sqrt(8/27)) < 1e-6
    @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
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
        @test abs(r.minimum - sqrt(8/27)) < 1e-6
        @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
    end
    @testset "Infinite lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [10, 10])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = NLoptAlg(:LD_MMA)
        r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
        @info "Status: $(r.status)"
        @test abs(r.minimum - sqrt(8/27)) < 1e-6
        @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
    end
    @testset "Infinite upper and lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [Inf, Inf])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = NLoptAlg(:LD_MMA)
        r = NonconvexCore.optimize(m, alg, [1.234, 2.345], options = options)
        @info "Status: $(r.status)"
        @test abs(r.minimum - sqrt(8/27)) < 1e-6
        @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
    end
end
