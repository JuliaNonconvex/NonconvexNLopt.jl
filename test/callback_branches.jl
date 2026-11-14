@testitem "Constraint callbacks at fresh points" begin
    using NonconvexNLopt, NonconvexCore, LinearAlgebra, Test

    # Gradient-based NLopt algorithms evaluate the constraint callbacks at
    # points that differ from the cached one during the solve, exercising
    # the fresh-point branches of the ineq/eq callback closures.
    m = NonconvexCore.Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> x[1]^2 + x[2]^2 - 1.0)
    for alg in (:LD_MMA, :LD_SLSQP)
        r = NonconvexNLopt.optimize(
            m, NLoptAlg(alg), [2.0, 2.0]; options=NLoptOptions(maxeval=200)
        )
        @test isfinite(r.minimum)
    end
    # Equality constraints are supported by SLSQP, COBYLA and ISRES.
    m2 = NonconvexCore.Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m2, x -> x[1]^2 + x[2]^2 - 1.0)
    add_eq_constraint!(m2, x -> x[1]^2 + x[2] - 1.5)
    for alg in (:LD_SLSQP, :LN_COBYLA, :GN_ISRES)
        r = NonconvexNLopt.optimize(
            m2, NLoptAlg(alg), [2.0, 2.0]; options=NLoptOptions(maxeval=200)
        )
        @test isfinite(r.minimum)
    end
end
