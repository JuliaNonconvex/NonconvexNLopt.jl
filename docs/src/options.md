# Options

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
