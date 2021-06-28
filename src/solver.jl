using Parameters
using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots
using QuantEcon

struct solution
    resultmessage::String
    F::Matrix{Float64}
    Q::Any
    xss::Vector{Float64}

    A::Matrix{Float64} 
    B::Matrix{Float64}
    C::Matrix{Float64}
    E::Any
end


function ss_residual(x, equations, ϵ_sd, fargs)
    resid = equations(x, x, x, zero(ϵ_sd), ϵ_sd, fargs...)
    return resid
end


function get_ss(equations, xss_init, ϵ_sd, fargs)

    function ss_resid_nlsolve!(x)
        x = ss_residual(x, equations, ϵ_sd, fargs)
    end

    solver_result = nlsolve(ss_resid_nlsolve!, xss_init)
    return solver_result.zero

end


function rendahl_coeffs(equations, xss, shocks_ss, fargs)

    A = ForwardDiff.jacobian(t -> equations(t, xss, xss, 0.0, fargs...), xss)
    B = ForwardDiff.jacobian(t -> equations(xss, t, xss, 0.0, fargs...), xss)
    C = ForwardDiff.jacobian(t -> equations(xss, xss, t, 0.0, fargs...), xss)
    E = ForwardDiff.jacobian(t -> equations(xss, xss, xss, t, fargs...), shocks_ss)

    return A, B, C, E

end


function solve_system(A, B, C, E, maxiter=1000, tol=1e-6)
    
    #==
    Solves for P and Q using Rehndal's Algorithm
    ==#
    
    F0 = zero(A)
    S0 = zero(A)
    error = one(tol) + tol
    iter = 0
    
    while error > tol && iter <= maxiter
        
        F1 = -(C * F0 + B) \ A
        S1 = -(A * S0 + B) \ C
        
        error = maximum(C * F1 * F1  + B * F1 + A)
        
        F0 .= F1
        S0 .= S1
        
        iter += 1
        
    end
    
    XP = LinearAlgebra.eigen(F0)
    XS = LinearAlgebra.eigen(S0)
    
    if iter == maxiter
        outmessage = "Convergence Failed. Max Iterations Reached. Error: $error"
    elseif maximum(abs.(XP.values)) > 1.0
        outmessage = "No Stable Solution Exists!"
    elseif maximum(abs.(XS.values)) > 1.0
        outmessage = "Multiple Solutions Exist!"
    else
        outmessage = "Convergence Successful!"
    end
    
    Q = -(C * F + B) \ E

    println(outmessage)

    return F0, Q, A, B, C, outmessage

    
end




function solve(equations, fargs; xinit)


    xss = get_ss(equations, xinit, fargs)
    A, B, C, E = rendahl_coeffs(equations, xss, shocks_ss, fargs)
    F, Q, A, B, C, outmessage = solve_system(A, B, C, E)

    out = solution(outmessage, F, Q, xss, A, B, C, E)

    return out
    
end 

function simulate(sol::solution, timeperiods::Int64)

    F = sol.F
    Q = sol.Q
    G = Matrix(I, size(F, 1), size(F, 1))

    lss = LSS(F, Q, G)

    X_simul, _ = simulate(lss, timeperiods)

    return X_simul

end



function compute_irfs(sol::solution; timeperiods = 40)

    IRF = zeros(size(sol.F, 1), timeperiods)

    IRF[:, 1] = sol.Q
    for t in 2:timeperiods
        IRF[:, t] = sol.F * IRF[:, t-1]
    end

    IRF = IRF .+ sol.xss

    return IRF

end



function draw_irf(irf, sol::solution, varnames)

    xss = sol.xss
    @assert length(varnames) == size(irf)[1]
    n = length(varnames)
    timeperiods = size(irf)[2]

    plotlist = []
    for i in 1:n
        plottmp =  plot(1:timeperiods, irf[i, :], lw = 2, title = varnames[i])
        plottmp =  plot!(1:timeperiods, fill(xss[i], timeperiods), color = :black, linestyle = :dot)
        push!(plotlist, plottmp)
    end

    plot(plotlist..., legend = false, titlefont = font(10, "Arial"))

end