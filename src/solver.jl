using Parameters
using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots

struct solution
    resultmessage::String
    F::Matrix{Float64}
    Q::Matrix{Float64} 
    irf::Matrix{Float64} 
    xss::Vector{Float64}

    A::Matrix{Float64} 
    B::Matrix{Float64}
    C::Matrix{Float64}
    E::Matrix{Float64}
end


function get_ss(equations, xss_init, fargs)
    

    function ss_residual!(xss_init)
        xss_init = equations(xss_init, xss_init, xss_init, fargs...)
    end

    solver_result = nlsolve(ss_residual!, xss_init)

    return solver_result.zero

end


function ss_residual!(equations, xss_init)
    xss_init = equations(xss_init, xss_init, xss_init, [0.0], params)
end

function get_ss(equations, xss_init, fargs)
    

    function ss_residual!(xss_init)
        xss_init = equations(xss_init, xss_init, xss_init, [0.0], fargs...)
    end

    solver_result = nlsolve(ss_residual!, xss_init)

    return solver_result.zero

end


function rendahl_coeffs(equations, xss, fargs)

    A = ForwardDiff.jacobian(t -> equations(t, xss, xss, 0.0, fargs...), xss)
    B = ForwardDiff.jacobian(t -> equations(xss, t, xss, 0.0, fargs...), xss)
    C = ForwardDiff.jacobian(t -> equations(xss, xss, t, 0.0, fargs...), xss)

    return A, B, C

end


function solve_system(A, B, C, maxiter=1000, tol=1e-6)
    
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
        outmessage = "Non existence"
    elseif maximum(abs.(XS.values)) > 1.0
        outmessage = "No stable equilibrium"
    else
        outmessage = "Convergence Successful!"
    end
    
    println(outmessage)

    return F0, A, B, C, outmessage

    
end


function compute_irfs(equations, F, B, C, shocks, xss, fargs, timeperiods = 40)

    shocks_ss = [zero(shocks)]
    E = ForwardDiff.jacobian(t -> equations(xss, xss, xss, shocks * t, fargs...), shocks_ss)
    Q = -(C * F + B) \ E

    IRF = zeros(size(F, 1), timeperiods)

    IRF[:, 1] = Q
    for t in 2:timeperiods
        IRF[:, t] = F * IRF[:, t-1]
    end

    IRF = IRF .+ xss

    return IRF, Q, E

end



function draw_irf(irf, xss, varnames)

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


function solve(equations, fargs, shocks; xinit, irf_timeperiods)


    xss = get_ss(equations, xinit, fargs)
    A, B, C = rendahl_coeffs(equations, xss, fargs)
    F, A, B, C, outmessage = solve_system(A, B, C)
    irf, Q, E = compute_irfs(equations, F, B, C, shocks, xss, fargs, irf_timeperiods)

    out = solution(outmessage, F, Q, irf, xss, A, B, C, E)

    return out
    
end 
