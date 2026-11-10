# Additional coverage tests for NonconvexNLopt.jl (branch feat/capabilities).
#
# These @testitem blocks focus on code paths that the main suite does not
# exercise directly: algorithm-name validation (near-miss and generic error
# branches), the per-algorithm `solver_capabilities` table, the
# `_meta_capabilities` fallbacks, `NLoptOptions` merging and option/suboption
# forwarding into the NLopt problems, the caching callback machinery, and the
# fields of `NLoptResult`.

@testitem "coverage: NLoptAlg constructors and algorithm-name validation" begin
    using NonconvexNLopt, Test

    # --- 1-arg and 2-arg constructors ---
    a = NLoptAlg(:LD_LBFGS)
    @test a isa NonconvexNLopt.NLoptAlg{Nothing}
    @test a.algorithm === :LD_LBFGS
    @test a.local_optimizer === nothing

    b = NLoptAlg(:AUGLAG, :LD_LBFGS)
    @test b isa NonconvexNLopt.NLoptAlg{Symbol}
    @test b.algorithm === :AUGLAG
    @test b.local_optimizer === :LD_LBFGS

    # A plain algorithm may carry a local optimizer without being rejected.
    c = NLoptAlg(:LD_LBFGS, :LN_BOBYQA)
    @test c.algorithm === :LD_LBFGS
    @test c.local_optimizer === :LN_BOBYQA

    # --- every meta-algorithm requires a local optimizer ---
    for alg in (:AUGLAG, :AUGLAG_EQ, :G_MLSL, :G_MLSL_LDS)
        err = try
            NLoptAlg(alg)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("A meta-algorithm", err.msg)
        @test occursin("no local optimizer was specified", err.msg)
        @test occursin("local_algorithm", err.msg)
    end

    # --- near-miss names (Levenshtein distance < 4) hit the "Did you mean" branch ---
    err = try
        NLoptAlg(:LD_MMM)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Did you mean :LD_MMA?", err.msg)

    err = try
        NLoptAlg(:GN_DIREC)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Did you mean :GN_DIRECT?", err.msg)

    # --- completely unrelated names hit the "valid algorithms are" branch ---
    err = try
        NLoptAlg(:NOT_AN_ALGORITHM)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("The valid algorithms are", err.msg)
    @test !occursin("Did you mean", err.msg)

    # Boundary case: distance exactly 4 does not trigger the suggestion branch.
    err = try
        NLoptAlg(:AU)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("The valid algorithms are", err.msg)
    @test !occursin("Did you mean", err.msg)

    # --- invalid local optimizers ---
    err = try
        NLoptAlg(:AUGLAG, :LD_TNEWTON_PRECON)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Did you mean :LD_TNEWTON_PRECOND?", err.msg)

    # Meta-algorithms are not valid local optimizers (generic branch).
    err = try
        NLoptAlg(:AUGLAG, :G_MLSL)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("The valid algorithms are", err.msg)
    @test !occursin("Did you mean", err.msg)

    # ...also for a plain outer algorithm.
    @test_throws ArgumentError NLoptAlg(:LD_LBFGS, :AUGLAG)
    @test_throws ArgumentError NLoptAlg(:LN_COBYLA, :AUGLAG_EQ)
    @test_throws ArgumentError NLoptAlg(:AUGLAG, :NOT_AN_ALGORITHM)

    # --- internal validation helpers ---
    @test NonconvexNLopt.__check_alg(:LD_LBFGS) === nothing
    @test NonconvexNLopt.__check_alg(:LD_LBFGS; loc=true) === nothing
    @test NonconvexNLopt.__check_alg(:AUGLAG) === nothing # meta names are in ALL_ALGORITHMS
    @test_throws ArgumentError NonconvexNLopt.__check_alg(:AUGLAG; loc=true)
    @test_throws ArgumentError NonconvexNLopt.__check_alg(:BOGUS)
    @test_throws ArgumentError NonconvexNLopt.__check_alg(:BOGUS; loc=true)

    @test NonconvexNLopt._check_alg(:LD_LBFGS, nothing) === nothing
    @test NonconvexNLopt._check_alg(:LD_LBFGS, :LN_COBYLA) === nothing
    @test NonconvexNLopt._check_alg(:AUGLAG, :LD_LBFGS) === nothing
    @test_throws ArgumentError NonconvexNLopt._check_alg(:AUGLAG, nothing)
    @test_throws ArgumentError NonconvexNLopt._check_alg(:AUGLAG_EQ, nothing)
    @test_throws ArgumentError NonconvexNLopt._check_alg(:AUGLAG, :G_MLSL)
    @test_throws ArgumentError NonconvexNLopt._check_alg(:BOGUS, :LD_LBFGS)
    @test_throws ArgumentError NonconvexNLopt._check_alg(:AUGLAG, :BOGUS)

    # --- is_zero_order dispatch (both methods) ---
    @test NonconvexNLopt.is_zero_order(:LN_NEWUOA, nothing) === true
    @test NonconvexNLopt.is_zero_order(:LD_LBFGS, nothing) === false
    @test NonconvexNLopt.is_zero_order(:AUGLAG, :LN_NELDERMEAD) === true
    @test NonconvexNLopt.is_zero_order(:AUGLAG, :LD_LBFGS) === false
end

@testitem "coverage: solver_capabilities for every algorithm family" begin
    using NonconvexNLopt, Test

    # Full declaration for a plain NLopt algorithm: never semidefinite, never
    # discrete, always continuous; the variable fields are eq/ineq/box/bounds.
    function expected_caps(;
        linear_eq, nonlinear_eq, linear_ineq, nonlinear_ineq, box, infinite_bounds
    )
        return ConstraintCapabilities(
            linear_eq,
            nonlinear_eq,
            linear_ineq,
            nonlinear_ineq,
            false, # linear_sd
            false, # nonlinear_sd
            box,
            true,  # continuous
            false, # binary
            false, # integer
            infinite_bounds,
        )
    end

    # Expected capabilities for every non-meta algorithm, and for every
    # meta-algorithm paired with :LD_LBFGS as local optimizer.
    entries = [
        # global derivative-free, no constraints, finite bounds required
        :GN_DIRECT => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GN_DIRECT_L => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GNL_DIRECT_NOSCAL => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GN_DIRECT_L_NOSCAL => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GN_DIRECT_L_RAND_NOSCAL => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GN_ORIG_DIRECT => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=false,
        ),
        :GN_ORIG_DIRECT_L => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=false,
        ),
        :GN_CRS2_LM => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GN_AGS => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=false,
        ),
        :GN_ESCH => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GN_ISRES => expected_caps(
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=false,
        ),
        :GD_STOGO => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :GD_STOGO_RAND => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        # local derivative-free
        :LN_COBYLA => expected_caps(
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=true,
        ),
        :LN_BOBYQA => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LN_NEWUOA => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=false,
            infinite_bounds=true,
        ),
        :LN_NEWUOA_BOUND => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LN_PRAXIS => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LN_NELDERMEAD => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LN_SBPLX => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        # local gradient-based
        :LD_CCSAQ => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=true,
        ),
        :LD_MMA => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=true,
        ),
        :LD_SLSQP => expected_caps(
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=true,
        ),
        :LD_LBFGS => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LD_TNEWTON => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LD_TNEWTON_PRECOND => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LD_TNEWTON_RESTART => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LD_TNEWTON_PRECOND_RESTART => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LD_VAR1 => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        :LD_VAR2 => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
        # meta-algorithms with :LD_LBFGS as local optimizer
        :G_MLSL => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :G_MLSL_LDS => expected_caps(
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=false,
        ),
        :AUGLAG => expected_caps(
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=true,
            nonlinear_ineq=true,
            box=true,
            infinite_bounds=true,
        ),
        :AUGLAG_EQ => expected_caps(
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=false,
            nonlinear_ineq=false,
            box=true,
            infinite_bounds=true,
        ),
    ]

    for (alg, expected) in entries
        loc = alg in NonconvexNLopt.META_ALGORITHMS ? :LD_LBFGS : nothing
        caps = solver_capabilities(NLoptAlg(alg, loc))
        @test caps isa ConstraintCapabilities
        @test caps == expected
    end

    # No capability is ever declared `missing`.
    for a in NonconvexNLopt.ALL_ALGORITHMS
        loc = a in NonconvexNLopt.META_ALGORITHMS ? :LD_LBFGS : nothing
        caps = solver_capabilities(NLoptAlg(a, loc))
        for f in fieldnames(ConstraintCapabilities)
            @test getfield(caps, f) isa Bool
        end
    end

    # --- meta-algorithms inherit the finite-bounds requirement of the local ---
    @test solver_capabilities(NLoptAlg(:AUGLAG, :GN_DIRECT)).infinite_bounds === false
    @test solver_capabilities(NLoptAlg(:AUGLAG, :GD_STOGO)).infinite_bounds === false
    @test solver_capabilities(NLoptAlg(:AUGLAG, :GD_STOGO_RAND)).infinite_bounds === false
    @test solver_capabilities(NLoptAlg(:AUGLAG, :LN_COBYLA)).infinite_bounds === true
    @test solver_capabilities(NLoptAlg(:AUGLAG, :LD_SLSQP)).infinite_bounds === true

    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :GN_DIRECT)).infinite_bounds === false
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :GN_DIRECT)).nonlinear_ineq === false
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :GN_ISRES)).infinite_bounds === false

    # --- AUGLAG_EQ inequality support follows the local optimizer ---
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :LD_LBFGS)).nonlinear_ineq === false
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :LN_COBYLA)).nonlinear_ineq === true
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :LN_COBYLA)).linear_ineq === true
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :LD_MMA)).nonlinear_ineq === true
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :LD_CCSAQ)).nonlinear_ineq === true
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :GN_AGS)).nonlinear_ineq === true
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :GN_AGS)).infinite_bounds === false
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :GN_ORIG_DIRECT)).nonlinear_ineq === true
    @test solver_capabilities(NLoptAlg(:AUGLAG_EQ, :GN_ORIG_DIRECT_L)).nonlinear_ineq ===
        true

    # --- MLSL variants declare box-only problems, whatever the local solver ---
    for loc_alg in (:LN_BOBYQA, :LD_LBFGS, :LN_COBYLA)
        for meta in (:G_MLSL, :G_MLSL_LDS)
            caps = solver_capabilities(NLoptAlg(meta, loc_alg))
            @test caps == expected_caps(
                linear_eq=false,
                nonlinear_eq=false,
                linear_ineq=false,
                nonlinear_ineq=false,
                box=true,
                infinite_bounds=false,
            )
        end
    end

    # --- algorithm-set constants used by the capability machinery ---
    @test NonconvexNLopt.META_ALGORITHMS == (:G_MLSL, :G_MLSL_LDS, :AUGLAG, :AUGLAG_EQ)
    @test NonconvexNLopt.UNCONSTRAINED_ALGORITHMS == (:LN_NEWUOA,)
    @test :G_MLSL in NonconvexNLopt.FINITE_BOUNDS_ALGORITHMS
    @test :G_MLSL_LDS in NonconvexNLopt.FINITE_BOUNDS_ALGORITHMS
    @test !(:AUGLAG in NonconvexNLopt.FINITE_BOUNDS_ALGORITHMS)
    @test :LD_SLSQP in NonconvexNLopt.EQUALITY_ALGORITHMS
    @test :LN_COBYLA in NonconvexNLopt.EQUALITY_ALGORITHMS
    @test :GN_ISRES in NonconvexNLopt.EQUALITY_ALGORITHMS
    @test !(:GN_DIRECT in NonconvexNLopt.INEQUALITY_ALGORITHMS)
    @test :GN_ORIG_DIRECT in NonconvexNLopt.INEQUALITY_ALGORITHMS
    @test Set(NonconvexNLopt.ALGORITHMS) == Set((
        NonconvexNLopt.ZERO_ORDER_ALGORITHMS..., NonconvexNLopt.FIRST_ORDER_ALGORITHMS...
    ))
end

@testitem "coverage: _meta_capabilities fallbacks with a missing local optimizer" begin
    using NonconvexNLopt, Test

    # The public constructor rejects meta-algorithms without a local optimizer,
    # but `_meta_capabilities` still carries `loc === nothing` fallback branches.
    # Reach them by temporarily relaxing the constructor guard, and restore the
    # original guard afterwards so that other test items are unaffected.
    Core.eval(
        NonconvexNLopt, :(function NonconvexNLopt._check_alg(alg::Symbol, loc::Nothing)
            return nothing
        end)
    )
    try
        alg_auglag = NLoptAlg(:AUGLAG, nothing)
        caps = NonconvexNLopt._meta_capabilities(alg_auglag)
        @test caps isa ConstraintCapabilities
        @test caps.linear_eq === true
        @test caps.nonlinear_eq === true
        @test caps.linear_ineq === true
        @test caps.nonlinear_ineq === true
        @test caps.box === true
        @test caps.infinite_bounds === true
        @test caps.continuous === true
        @test caps.binary === false
        @test caps.integer === false

        alg_auglag_eq = NLoptAlg(:AUGLAG_EQ, nothing)
        caps = NonconvexNLopt._meta_capabilities(alg_auglag_eq)
        @test caps.linear_eq === true
        @test caps.nonlinear_eq === true
        @test caps.linear_ineq === false
        @test caps.nonlinear_ineq === false
        @test caps.box === true
        @test caps.infinite_bounds === true

        # `solver_capabilities` routes these through `_meta_capabilities` as well.
        caps = NonconvexNLopt.solver_capabilities(alg_auglag)
        @test caps.infinite_bounds === true
        caps = NonconvexNLopt.solver_capabilities(alg_auglag_eq)
        @test caps.nonlinear_ineq === false
    finally
        # Remove the relaxed guard method, restoring the strict constructor.
        Base.delete_method(which(NonconvexNLopt._check_alg, Tuple{Symbol,Nothing}))
    end

    # The strict guard is back: meta-algorithms without a local optimizer error.
    @test_throws ArgumentError NLoptAlg(:AUGLAG, nothing)
    @test_throws ArgumentError NLoptAlg(:AUGLAG_EQ, nothing)
end

@testitem "coverage: NLoptOptions construction, merging, and forwarding" begin
    using NonconvexNLopt, NonconvexCore, Test, NLopt

    # --- defaults ---
    o = NLoptOptions()
    @test o.nt.ftol_rel == 1e-6
    @test o.nt.ftol_abs == 1e-6
    @test o.nt.xtol_rel == 1e-6
    @test o.nt.xtol_abs == 1e-6
    @test o.nt.suboptions isa NLoptOptions
    @test isempty(o.nt.suboptions.nt)

    # --- extra kwargs are merged, defaults can be overridden ---
    o2 = NLoptOptions(ftol_rel=1e-9, maxeval=100)
    @test o2.nt.ftol_rel == 1e-9
    @test o2.nt.maxeval == 100
    @test o2.nt.xtol_rel == 1e-6
    @test o2.nt.suboptions isa NLoptOptions

    o3 = NLoptOptions(suboptions=NLoptOptions(xtol_rel=1e-12))
    @test o3.nt.suboptions.nt.xtol_rel == 1e-12

    # --- a model to exercise get_nlopt_problem ---
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-1.0, -1.0], [1.0, 1.0])
    x0 = [0.5, 0.5]
    vm, vx0, _ = NonconvexCore.tovecmodel(m, x0)

    # --- options forwarded onto the top-level problem ---
    opts = NLoptOptions(ftol_rel=1e-7, xtol_rel=1e-7, maxeval=1234)
    problem, counter = NonconvexNLopt.get_nlopt_problem(:LD_LBFGS, nothing, opts, vm, vx0)
    @test problem isa NLopt.Opt
    @test string(problem.algorithm) == "LD_LBFGS"
    @test problem.ftol_rel == 1e-7
    @test problem.xtol_rel == 1e-7
    @test problem.maxeval == 1234
    @test problem.lower_bounds == [-1.0, -1.0]
    @test problem.upper_bounds == [1.0, 1.0]
    # The initial cached evaluation goes through the CountingFunction.
    @test counter[] == 1

    # --- unknown option names fail when forwarded to the NLopt problem ---
    err = try
        NonconvexNLopt.Workspace(
            vm, NLoptAlg(:LD_LBFGS), vx0; options=NLoptOptions(not_an_nlopt_option=1)
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("no writable property", err.msg)

    # --- suboptions are forwarded to the local optimizer of meta-algorithms ---
    subopts = NLoptOptions(suboptions=NLoptOptions(xtol_rel=1e-6))
    problem2, counter2 = NonconvexNLopt.get_nlopt_problem(
        :AUGLAG, :LD_LBFGS, subopts, vm, vx0
    )
    @test problem2 isa NLopt.Opt
    @test string(problem2.algorithm) == "AUGLAG"
    @test counter2[] == 1

    # An unknown *suboption* also fails, but only for meta-algorithms.
    err = try
        NonconvexNLopt.Workspace(
            vm,
            NLoptAlg(:AUGLAG, :LD_LBFGS),
            vx0;
            options=NLoptOptions(suboptions=NLoptOptions(not_an_nlopt_option=1)),
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("no writable property", err.msg)

    # For a plain algorithm no subproblem exists, so the suboptions are
    # ignored (documenting current behavior).
    ws = NonconvexNLopt.Workspace(
        vm,
        NLoptAlg(:LD_LBFGS),
        vx0;
        options=NLoptOptions(suboptions=NLoptOptions(not_an_nlopt_option=1)),
    )
    @test ws.problem isa NLopt.Opt

    # Nested suboptions hit the `_k !== :suboptions` guard and are skipped.
    ws = NonconvexNLopt.Workspace(
        vm,
        NLoptAlg(:AUGLAG, :LD_LBFGS),
        vx0;
        options=NLoptOptions(
            suboptions=NLoptOptions(suboptions=NLoptOptions(xtol_rel=1e-4))
        ),
    )
    @test ws.problem isa NLopt.Opt

    # --- get_nlopt_problem on the VecModel (constraint collections) ---
    m2 = Model(x -> sum(x .^ 2))
    addvar!(m2, [-1.0, -1.0], [1.0, 1.0])
    add_ineq_constraint!(m2, x -> sum(x) - 0.5)
    add_eq_constraint!(m2, x -> x[1] - 2 * x[2])
    vm2, vx02, _ = NonconvexCore.tovecmodel(m2, x0)
    problem3, _ = NonconvexNLopt.get_nlopt_problem(
        :LD_SLSQP, nothing, NLoptOptions(), vm2, vx02
    )
    @test problem3 isa NLopt.Opt
    @test string(problem3.algorithm) == "LD_SLSQP"
end

@testitem "coverage: optimize! results, counters, and representative runs" begin
    using NonconvexNLopt, NonconvexCore, Test, LinearAlgebra, NLopt

    # --- NLoptResult fields, via the public optimize entry point ---
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-10.0, -10.0], [10.0, 10.0])
    options = NLoptOptions(ftol_rel=1e-9, xtol_rel=1e-9, maxeval=5000)
    alg = NLoptAlg(:LD_LBFGS)
    r = NonconvexCore.optimize(m, alg, [5.0, 5.0]; options=options)
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
    @test norm(r.minimizer) < 1e-6

    # --- optimize without an explicit initial point (getinit default) ---
    # The default argument `x0 = getinit(model)` in `NLoptWorkspace` relies on
    # `getinit` being in scope inside NonconvexNLopt; on NonconvexCore 2.0.0
    # it is not reexported, so this path currently raises UndefVarError. Run
    # the check only once the binding exists (and report the finding).
    if isdefined(NonconvexNLopt, :getinit)
        m_init = Model(x -> sum(x .^ 2))
        addvar!(m_init, [-2.0, -2.0], [2.0, 2.0]; init=[0.9, 0.9])
        r_init = NonconvexCore.optimize(m_init, NLoptAlg(:LD_LBFGS); options=options)
        @test abs(r_init.minimum) < 1e-6
    end

    # --- Workspace construction and reuse: the counter resets per run ---
    vm, vx0, _ = NonconvexCore.tovecmodel(m, [5.0, 5.0])
    ws = NonconvexNLopt.Workspace(vm, alg, vx0; options=options)
    @test ws isa NonconvexNLopt.NLoptWorkspace
    @test ws.problem isa NLopt.Opt
    @test ws.x0 == vx0
    @test ws.alg === alg
    @test ws.options === options
    r1 = NonconvexNLopt.optimize!(ws)
    @test r1.fcalls >= 1
    @test r1.minimum < 1e-6
    ws.counter[] = 9999
    r2 = NonconvexNLopt.optimize!(ws)
    @test r2.fcalls >= 1
    @test r2.minimum < 1e-6

    # Direct NLoptWorkspace construction, default initial point (same guard as
    # above: the default argument needs `getinit` in NonconvexNLopt's scope).
    if isdefined(NonconvexNLopt, :getinit)
        ws2 = NonconvexNLopt.NLoptWorkspace(
            NonconvexCore.tovecmodel(m_init)[1], alg; options=options
        )
        @test ws2.x0 == [0.9, 0.9]
        @test ws2.counter[] == 1 # initial cached evaluation
    end

    # --- zero-order algorithm (initial cache update without gradients) ---
    r_zo = NonconvexCore.optimize(
        m,
        NLoptAlg(:LN_NELDERMEAD),
        [1.0, 1.0];
        options=NLoptOptions(xtol_rel=1e-6, maxeval=2000),
    )
    @test abs(r_zo.minimum) < 1e-4

    # --- unconstrained-only algorithm with infinite bounds ---
    mu = Model(x -> sum(x .^ 2))
    addvar!(mu, [-Inf, -Inf], [Inf, Inf]; init=[1.0, 1.0])
    r_nw = NonconvexCore.optimize(
        mu,
        NLoptAlg(:LN_NEWUOA),
        [1.0, 1.0];
        options=NLoptOptions(xtol_rel=1e-8, maxeval=500),
    )
    @test abs(r_nw.minimum) < 1e-6

    # --- constant objective: Zygote gradient is `nothing` -> zeros fallback ---
    mc = Model(x -> 2.0)
    addvar!(mc, [-1.0, -1.0], [1.0, 1.0])
    r_c = NonconvexCore.optimize(
        mc, NLoptAlg(:LD_LBFGS), [0.5, 0.5]; options=NLoptOptions(maxeval=100)
    )
    @test r_c.minimum ≈ 2.0
    @test r_c.status isa Symbol

    # --- inequality + equality constraints (first-order jacobian machinery) ---
    f(x) = x[2] >= 0 ? sqrt(x[2]) : Inf
    g(x, a, b) = (a * x[1] + b)^3 - x[2]
    mcon = Model(f)
    addvar!(mcon, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(mcon, x -> g(x, 2, 0))
    add_ineq_constraint!(mcon, x -> g(x, -1, 1))
    add_eq_constraint!(mcon, x -> sum(x) - 1 / 3 - 8 / 27)
    r_con = NonconvexCore.optimize(
        mcon,
        NLoptAlg(:LD_SLSQP),
        [1.234, 2.345];
        options=NLoptOptions(xtol_rel=1e-6, maxeval=5000),
    )
    @test abs(r_con.minimum - sqrt(8 / 27)) < 1e-6

    # --- zero-order solver with inequality constraints (shared cache) ---
    mzo = Model(f)
    addvar!(mzo, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(mzo, x -> g(x, 2, 0))
    add_ineq_constraint!(mzo, x -> g(x, -1, 1))
    r_cob = NonconvexCore.optimize(
        mzo,
        NLoptAlg(:LN_COBYLA),
        [1.234, 2.345];
        options=NLoptOptions(xtol_rel=1e-6, maxeval=5000),
    )
    @test abs(r_cob.minimum - sqrt(8 / 27)) < 1e-4

    # --- block (vector-valued) constraints exercise the per-row jacobian loop ---
    mblk = Model(f)
    addvar!(mblk, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(mblk, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))
    r_blk = NonconvexCore.optimize(
        mblk,
        NLoptAlg(:LD_MMA),
        [1.234, 2.345];
        options=NLoptOptions(xtol_rel=1e-6, maxeval=5000),
    )
    @test abs(r_blk.minimum - sqrt(8 / 27)) < 1e-6

    # --- constant constraint: jacobian row is `nothing` -> zero-row fallback ---
    mcc = Model(x -> sum(x .^ 2))
    addvar!(mcc, [-1.0, -1.0], [1.0, 1.0])
    add_ineq_constraint!(mcc, x -> -1.0)
    r_cc = NonconvexCore.optimize(
        mcc,
        NLoptAlg(:LD_MMA),
        [0.5, 0.5];
        options=NLoptOptions(xtol_rel=1e-8, maxeval=2000),
    )
    @test abs(r_cc.minimum) < 1e-6

    # --- meta-algorithms: AUGLAG with gradient-based local optimizer ---
    maug = Model(x -> sqrt(x[2]) + x[1]^2)
    addvar!(maug, [0.0, 0.0], [10.0, 10.0])
    r_aug = NonconvexCore.optimize(
        maug,
        NLoptAlg(:AUGLAG, :LD_LBFGS),
        [0.1, 0.2];
        options=NLoptOptions(
            xtol_rel=1e-6,
            ftol_rel=1e-6,
            maxeval=1000,
            suboptions=NLoptOptions(xtol_rel=1e-6),
        ),
    )
    @test abs(r_aug.minimum) < 1e-6

    # --- meta-algorithms: MLSL (finite bounds) with a zero-order local ---
    # Note: with :LN_BOBYQA as the MLSL local optimizer, NLopt stops at the
    # start point with ROUNDOFF_LIMITED under the default ftol/xtol options;
    # :LN_COBYLA converges.
    mmlsl = Model(x -> sum(x .^ 2))
    addvar!(mmlsl, [-1.0, -1.0], [1.0, 1.0])
    r_mlsl = NonconvexCore.optimize(
        mmlsl,
        NLoptAlg(:G_MLSL_LDS, :LN_COBYLA),
        [0.5, 0.5];
        options=NLoptOptions(xtol_rel=1e-4, maxeval=500, population=10),
    )
    @test abs(r_mlsl.minimum) < 1e-4

    # --- meta-algorithms: AUGLAG_EQ delegates inequalities to the local ---
    # AUGLAG_EQ penalizes the equality constraint and hands the inequalities to
    # the subsidiary algorithm, which must support them (:LD_MMA does).
    maug_eq = Model(f)
    addvar!(maug_eq, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(maug_eq, x -> g(x, 2, 0))
    add_ineq_constraint!(maug_eq, x -> g(x, -1, 1))
    add_eq_constraint!(maug_eq, x -> sum(x) - 1 / 3 - 8 / 27)
    r_aeq = NonconvexCore.optimize(
        maug_eq,
        NLoptAlg(:AUGLAG_EQ, :LD_MMA),
        [1.234, 2.345];
        options=NLoptOptions(xtol_rel=1e-6, maxeval=5000),
    )
    @test abs(r_aeq.minimum - sqrt(8 / 27)) < 1e-6
end

@testitem "coverage: capability violations rejected by NonconvexCore.optimize" begin
    using NonconvexNLopt, NonconvexCore, Test

    # Inequality constraints rejected by bound-only solvers.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-1.0, -1.0], [1.0, 1.0])
    add_ineq_constraint!(m, x -> sum(x) - 0.5)
    err = try
        NonconvexCore.optimize(m, NLoptAlg(:LN_BOBYQA), [0.5, 0.5])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("inequality", err.msg)

    # ...also through the MLSL meta-algorithm (bound-only problems).
    err = try
        NonconvexCore.optimize(m, NLoptAlg(:G_MLSL, :LN_BOBYQA), [0.5, 0.5])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError

    # Box bounds rejected by the unconstrained-only NEWUOA.
    m2 = Model(x -> sum(x .^ 2))
    addvar!(m2, [-1.0, -1.0], [1.0, 1.0])
    err = try
        NonconvexCore.optimize(m2, NLoptAlg(:LN_NEWUOA), [0.5, 0.5])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("bounds", err.msg)

    # Infinite bounds rejected by finite-bounds-only global algorithms.
    m3 = Model(x -> sum(x .^ 2))
    addvar!(m3, [-Inf, -Inf], [1.0, 1.0])
    for alg in (:GN_DIRECT, :GD_STOGO, :GN_AGS)
        err_infb = try
            NonconvexCore.optimize(m3, NLoptAlg(alg), [0.5, 0.5])
            nothing
        catch e
            e
        end
        @test err_infb isa ArgumentError
        @test occursin("infinite", err_infb.msg)
    end

    # Equality constraints rejected by solvers without equality support.
    m4 = Model(x -> sum(x .^ 2))
    addvar!(m4, [-1.0, -1.0], [1.0, 1.0])
    add_eq_constraint!(m4, x -> sum(x) - 0.5)
    err = try
        NonconvexCore.optimize(m4, NLoptAlg(:LD_MMA), [0.5, 0.5])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("equality", err.msg)

    # AUGLAG_EQ with a local optimizer that cannot handle inequalities.
    m5 = Model(x -> sum(x .^ 2))
    addvar!(m5, [-1.0, -1.0], [1.0, 1.0])
    add_ineq_constraint!(m5, x -> sum(x) - 0.5)
    err = try
        NonconvexCore.optimize(m5, NLoptAlg(:AUGLAG_EQ, :LD_LBFGS), [0.5, 0.5])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("inequality", err.msg)
end
