using JET, NonconvexNLopt

# Run JET analysis on the package.
JET.test_package(NonconvexNLopt; target_modules=(NonconvexNLopt,))
