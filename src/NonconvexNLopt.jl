module NonconvexNLopt

export NLoptAlg, NLoptOptions

using Reexport, Parameters, SparseArrays, Zygote
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
import NonconvexCore: optimize!, Workspace
import NLopt
import StringDistances

const META_ALGORITHMS = (:G_MLSL, :G_MLSL_LDS, :AUGLAG, :AUGLAG_EQ)

const ZERO_ORDER_ALGORITHMS = (
    :GN_DIRECT,
    :GN_DIRECT_L,
    :GNL_DIRECT_NOSCAL,
    :GN_DIRECT_L_NOSCAL,
    :GN_DIRECT_L_RAND_NOSCAL,
    :GN_ORIG_DIRECT,
    :GN_ORIG_DIRECT_L,
    :GN_CRS2_LM,
    :GN_AGS,
    :GN_ESCH,
    :GN_ISRES,
    :LN_COBYLA,
    :LN_BOBYQA,
    :LN_NEWUOA,
    :LN_NEWUOA_BOUND,
    :LN_PRAXIS,
    :LN_NELDERMEAD,
    :LN_SBPLX,
)

const FIRST_ORDER_ALGORITHMS = (
    :GD_STOGO,
    :GD_STOGO_RAND,
    :LD_CCSAQ,
    :LD_MMA,
    :LD_SLSQP,
    :LD_LBFGS,
    :LD_TNEWTON,
    :LD_TNEWTON_PRECOND,
    :LD_TNEWTON_RESTART,
    :LD_TNEWTON_PRECOND_RESTART,
    :LD_VAR1,
    :LD_VAR2,
)

const ALGORITHMS = (ZERO_ORDER_ALGORITHMS..., FIRST_ORDER_ALGORITHMS...)

# Algorithms that require finite bounds on every variable (see
# JuliaNonconvex/NonconvexCore.jl#33).
# Algorithms that require finite bounds on every variable. All of NLopt's
# global algorithms require bound constraints (NLopt documentation).
const FINITE_BOUNDS_ALGORITHMS = (
    :GN_DIRECT,
    :GN_DIRECT_L,
    :GNL_DIRECT_NOSCAL,
    :GN_DIRECT_L_NOSCAL,
    :GN_DIRECT_L_RAND_NOSCAL,
    :GN_ORIG_DIRECT,
    :GN_ORIG_DIRECT_L,
    :GN_CRS2_LM,
    :GN_AGS,
    :GN_ESCH,
    :GN_ISRES,
    :GD_STOGO,
    :GD_STOGO_RAND,
    :G_MLSL,
    :G_MLSL_LDS,
)

# Algorithms supporting nonlinear inequality constraints (NLopt documentation).
const INEQUALITY_ALGORITHMS = (
    :GN_ORIG_DIRECT,
    :GN_ORIG_DIRECT_L,
    :GN_AGS,
    :GN_ISRES,
    :LN_COBYLA,
    :LD_MMA,
    :LD_CCSAQ,
    :LD_SLSQP,
)

# Algorithms supporting nonlinear equality constraints (NLopt documentation).
const EQUALITY_ALGORITHMS = (:GN_ISRES, :LN_COBYLA, :LD_SLSQP)

# Algorithms that support no bound constraints at all (unconstrained only).
const UNCONSTRAINED_ALGORITHMS = (:LN_NEWUOA,)
const ALL_ALGORITHMS = (META_ALGORITHMS..., ALGORITHMS...)

function is_zero_order(algorithm, ::Nothing)
    return algorithm in ZERO_ORDER_ALGORITHMS
end
function is_zero_order(::Any, local_optimizer)
    return is_zero_order(local_optimizer, nothing)
end

function __check_alg(alg; loc = false)
    valid = loc ? ALGORITHMS : ALL_ALGORITHMS
    if !(alg in valid)
        dists =
            StringDistances.evaluate.(
                Ref(StringDistances.Levenshtein()),
                string(alg),
                string.(valid),
            )
        val, ind = findmin(dists)
        sim_alg = Symbol(valid[ind])
        if val < 4
            throw(
                ArgumentError(
                    "Algorithm $(Meta.quot(alg)) is not a valid algorithm. Did you mean $(Meta.quot(sim_alg))?",
                ),
            )
        else
            throw(
                ArgumentError(
                    "Algorithm $(Meta.quot(alg)) is not a valid algorithm. The valid algorithms are $valid.",
                ),
            )
        end
    end
    return nothing
end

function _check_alg(alg, loc)
    if alg in META_ALGORITHMS && loc === nothing
        throw(
            ArgumentError(
                "A meta-algorithm $(Meta.quot(alg)) was input but no local optimizer was specified. Please specify a local algorithm using `NLoptAlg($(Meta.quot(alg)), local_algorithm)` where `local_optimizer` is one of the following algorithms: $(ALGORITHMS).",
            ),
        )
    end
    __check_alg(alg, loc = false)
    loc === nothing || __check_alg(loc, loc = true)
    return nothing
end

struct NLoptAlg{L<:Union{Symbol,Nothing}} <: AbstractOptimizer
    algorithm::Symbol
    local_optimizer::L
    function NLoptAlg(alg::Symbol, loc::Union{Symbol,Nothing})
        _check_alg(alg, loc)
        return new{typeof(loc)}(alg, loc)
    end
end
NLoptAlg(alg) = NLoptAlg(alg, nothing)

# Declare the capabilities per NLopt's algorithm documentation: constraint
# support differs substantially between algorithms, and all global algorithms
# require finite bounds.
function NonconvexCore.solver_capabilities(alg::NLoptAlg)
    a = alg.algorithm
    if a in META_ALGORITHMS
        return _meta_capabilities(alg)
    end
    eq = a in EQUALITY_ALGORITHMS
    ineq = a in INEQUALITY_ALGORITHMS
    box = !(a in UNCONSTRAINED_ALGORITHMS)
    return ConstraintCapabilities(;
        linear_eq = eq,
        nonlinear_eq = eq,
        linear_ineq = ineq,
        nonlinear_ineq = ineq,
        linear_sd = false,
        nonlinear_sd = false,
        box = box,
        continuous = true,
        binary = false,
        integer = false,
        infinite_bounds = !(a in FINITE_BOUNDS_ALGORITHMS),
    )
end

function _meta_capabilities(alg::NLoptAlg)
    a = alg.algorithm
    loc = alg.local_optimizer
    if a in (:G_MLSL, :G_MLSL_LDS)
        # MLSL supports bound-constrained problems only, and requires finite
        # bounds (NLopt documentation).
        return ConstraintCapabilities(;
            linear_eq = false,
            nonlinear_eq = false,
            linear_ineq = false,
            nonlinear_ineq = false,
            linear_sd = false,
            nonlinear_sd = false,
            box = true,
            continuous = true,
            binary = false,
            integer = false,
            infinite_bounds = false,
        )
    elseif a === :AUGLAG
        # AUGLAG folds inequality and equality constraints into penalties and
        # passes an unconstrained subproblem to the local optimizer; the
        # finite-bounds requirement is inherited from the local optimizer.
        return ConstraintCapabilities(;
            linear_eq = true,
            nonlinear_eq = true,
            linear_ineq = true,
            nonlinear_ineq = true,
            linear_sd = false,
            nonlinear_sd = false,
            box = true,
            continuous = true,
            binary = false,
            integer = false,
            infinite_bounds = loc === nothing ? true : !(loc in FINITE_BOUNDS_ALGORITHMS),
        )
    else # :AUGLAG_EQ
        # AUGLAG_EQ penalizes equality constraints only; inequalities are
        # handled directly by the subsidiary algorithm.
        return ConstraintCapabilities(;
            linear_eq = true,
            nonlinear_eq = true,
            linear_ineq = loc === nothing ? false : loc in INEQUALITY_ALGORITHMS,
            nonlinear_ineq = loc === nothing ? false : loc in INEQUALITY_ALGORITHMS,
            linear_sd = false,
            nonlinear_sd = false,
            box = true,
            continuous = true,
            binary = false,
            integer = false,
            infinite_bounds = loc === nothing ? true : !(loc in FINITE_BOUNDS_ALGORITHMS),
        )
    end
end

struct NLoptOptions{N<:NamedTuple}
    nt::N
end
function NLoptOptions(;
    ftol_rel = 1e-6,
    ftol_abs = 1e-6,
    xtol_rel = 1e-6,
    xtol_abs = 1e-6,
    suboptions = NLoptOptions(NamedTuple()),
    kwargs...,
)
    NLoptOptions(
        merge(
            (; suboptions),
            NamedTuple(kwargs),
            (; ftol_rel, ftol_abs, xtol_rel, xtol_abs),
        ),
    )
end

mutable struct NLoptWorkspace{
    M<:VecModel,
    P,
    X<:AbstractVector,
    O<:NLoptOptions,
    A<:NLoptAlg,
} <: Workspace
    model::M
    problem::P
    x0::X
    options::O
    alg::A
    counter::Base.RefValue{Int}
end
function NLoptWorkspace(
    model::VecModel,
    optimizer::NLoptAlg,
    x0::AbstractVector = getinit(model);
    options = NLoptOptions(),
    kwargs...,
)
    problem, counter = get_nlopt_problem(
        optimizer.algorithm,
        optimizer.local_optimizer,
        options,
        model,
        copy(x0),
    )
    return NLoptWorkspace(model, problem, copy(x0), options, optimizer, counter)
end
struct NLoptResult{M1,M2,P,S,A,O} <: AbstractResult
    minimizer::M1
    minimum::M2
    problem::P
    status::S
    alg::A
    options::O
    fcalls::Int
end

function optimize!(workspace::NLoptWorkspace)
    @unpack problem, options, x0, counter = workspace
    counter[] = 0
    minf, minx, ret = NLopt.optimize(problem, x0)
    return NLoptResult(copy(minx), minf, problem, ret, workspace.alg, options, counter[])
end

function Workspace(model::VecModel, optimizer::NLoptAlg, args...; kwargs...)
    return NLoptWorkspace(model, optimizer, args...; kwargs...)
end

function get_nlopt_problem(
    algorithm,
    local_optimizer,
    options,
    model::VecModel,
    x0::AbstractVector,
)
    eq = if length(model.eq_constraints.fs) == 0
        nothing
    else
        model.eq_constraints
    end
    ineq = if length(model.ineq_constraints.fs) == 0
        nothing
    else
        model.ineq_constraints
    end
    obj = CountingFunction(getobjective(model))
    return get_nlopt_problem(
        algorithm,
        local_optimizer,
        options,
        obj,
        ineq,
        eq,
        x0,
        getmin(model),
        getmax(model),
    ),
    obj.counter
end
function get_nlopt_problem(
    algorithm,
    local_optimizer,
    options,
    obj,
    ineq_constr,
    eq_constr,
    x0,
    xlb,
    xub,
)
    local lastx, objval, objgrad
    local ineqconstrval, ineqconstrjac
    if ineq_constr !== nothing
        ineqconstrval, ineqconstrjac = nothing, nothing
    end
    local eqconstrval, eqconstrjac
    if eq_constr !== nothing
        eqconstrval, eqconstrjac = nothing, nothing
    end

    function update_cached_values(x, updategrad = true)
        lastx = copy(x)
        if updategrad
            objval, objpb = Zygote.pullback(obj, lastx)
            objgrad = objpb(1.0)[1]
            objgrad === nothing && (objgrad = zeros(length(lastx)))
        else
            objval = obj(lastx)
            # Invalidate the cached gradient: it belongs to the previous point,
            # and NLopt's meta-algorithms (AUGLAG, MLSL) evaluate the objective
            # and constraints without gradients between sub-solves.
            objgrad = nothing
        end
        if ineq_constr !== nothing
            if updategrad
                ineqconstrval, ineqconstrpb = Zygote.pullback(ineq_constr, lastx)
                ineqconstrjac = zeros(length(ineqconstrval), length(lastx))
                e = zeros(length(ineqconstrval))
                for i = 1:length(ineqconstrval)
                    e .= 0
                    e[i] = 1
                    row = ineqconstrpb(e)[1]
                    row === nothing || (ineqconstrjac[i, :] .= row)
                end
            else
                ineqconstrval = ineq_constr(lastx)
                ineqconstrjac = nothing
            end
        end
        if eq_constr !== nothing
            if updategrad
                eqconstrval, eqconstrpb = Zygote.pullback(eq_constr, lastx)
                eqconstrjac = zeros(length(eqconstrval), length(lastx))
                e = zeros(length(eqconstrval))
                for i = 1:length(eqconstrval)
                    e .= 0
                    e[i] = 1
                    row = eqconstrpb(e)[1]
                    row === nothing || (eqconstrjac[i, :] .= row)
                end
            else
                eqconstrval = eq_constr(lastx)
                eqconstrjac = nothing
            end
        end
    end
    update_cached_values(x0, !is_zero_order(algorithm, local_optimizer))

    function nlopt_obj(x, grad)
        if x == lastx
            if length(grad) > 0
                objgrad === nothing && update_cached_values(x, true)
                grad .= objgrad
            end
            return objval
        else
            update_cached_values(x, length(grad) > 0)
            if length(grad) > 0
                grad .= objgrad
            end
            return objval
        end
    end
    function nlopt_ineqconstr_gen(i)
        function (x, grad)
            if x == lastx
                if length(grad) > 0
                    ineqconstrjac === nothing && update_cached_values(x, true)
                    grad .= ineqconstrjac[i, :]
                end
                return ineqconstrval[i]
            else
                update_cached_values(x, length(grad) > 0)
                if length(grad) > 0
                    grad .= ineqconstrjac[i, :]
                end
                return ineqconstrval[i]
            end
        end
    end
    function nlopt_eqconstr_gen(i)
        function (x, grad)
            if x == lastx
                if length(grad) > 0
                    eqconstrjac === nothing && update_cached_values(x, true)
                    grad .= eqconstrjac[i, :]
                end
                return eqconstrval[i]
            else
                update_cached_values(x, length(grad) > 0)
                if length(grad) > 0
                    grad .= eqconstrjac[i, :]
                end
                return eqconstrval[i]
            end
        end
    end

    problem = NLopt.Opt(algorithm, length(x0))
    if local_optimizer !== nothing
        subproblem = NLopt.Opt(local_optimizer, length(x0))
        foreach(keys(options.nt.suboptions.nt)) do _k
            if _k !== :suboptions
                setproperty!(subproblem, _k, options.nt.suboptions.nt[_k])
            end
        end
        NLopt.local_optimizer!(problem, subproblem)
    end
    problem.lower_bounds = xlb
    problem.upper_bounds = xub
    problem.min_objective = nlopt_obj
    foreach(keys(options.nt)) do k
        if k != :suboptions
            setproperty!(problem, k, options.nt[k])
        end
    end

    if ineq_constr !== nothing
        for i = 1:length(ineqconstrval)
            NLopt.inequality_constraint!(problem, nlopt_ineqconstr_gen(i), 0.0)
        end
    end
    if eq_constr !== nothing
        for i = 1:length(eqconstrval)
            NLopt.equality_constraint!(problem, nlopt_eqconstr_gen(i), 0.0)
        end
    end

    return problem
end

end
