module NonconvexNLopt

export NLoptAlg, NLoptOptions

using Reexport, Parameters, SparseArrays, Zygote
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
import NonconvexCore: optimize!
using NLopt

include("nlopt.jl")

end
