@testitem "JET" begin
    using JET, NonconvexNLopt
    JET.test_package(NonconvexNLopt; target_modules=(NonconvexNLopt,))
end
