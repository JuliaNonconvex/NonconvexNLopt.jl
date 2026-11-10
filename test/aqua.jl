@testitem "Aqua" begin
    using Aqua, NonconvexNLopt
    Aqua.test_all(NonconvexNLopt)
end
