using Parameters
using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots

mutable struct solution
    F::Matrix{Float64}
    Q::Matrix{Float64} 
    irf::Matrix{Float64} 
    xss::Vector{Float64}

    A::Matrix{Float64} 
    B::Matrix{Float64}
    C::Matrix{Float64}
    E::Matrix{Float64}
end


function get_ss(F, xss_init, fargs)
    

    function ss_residual!(xss_init)
        xss_init = F(xss_init, xss_init, xss_init, fargs...)
    end

    solver_result = nlsolve(ss_residual!, xss_init)

    return solver_result.zero

end


function ss_residual!(xss_init)
    xss_init = F(xss_init, xss_init, xss_init, [0.0], params)
end

function get_ss(F, xss_init, fargs)
    

    function ss_residual!(xss_init)
        xss_init = F(xss_init, xss_init, xss_init, [0.0], fargs...)
    end

    solver_result = nlsolve(ss_residual!, xss_init)

    return solver_result.zero

end


function rendahl_coeffs(F, xss, fargs)

    A = ForwardDiff.jacobian(t -> F(t, xss, xss, 0.0, fargs...), xss)
    B = ForwardDiff.jacobian(t -> F(xss, t, xss, 0.0, fargs...), xss)
    C = ForwardDiff.jacobian(t -> F(xss, xss, t, 0.0, fargs...), xss)
    E = ForwardDiff.jacobian(t -> F(xss, xss, xss, t, fargs...), [0.0])

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
    
    Q = -(C * F0 + B) \ E
    
    if iter == maxiter
        println("Convergence Failed. Max Iterations Reached. Error: $error")
    elseif maximum(abs.(XP.values)) > 1.0
        println("Non existence")
    elseif maximum(abs.(XS.values)) > 1.0
        println("No stable equilibrium")
    else
        println("Convergence Successful!")
    end
    
    return F0, Q

    
end


function compute_irfs(F, Q, xss, timeperiods = 40)

    IRF = zeros(size(F, 1), timeperiods)

    IRF[:, 1] = Q
    for t in 2:timeperiods
        IRF[:, t] = F * IRF[:, t-1]
    end

    IRF = IRF .+ xss

    return IRF

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


function solve(F, fargs; xinit, irf_timeperiods)


    xss = get_ss(F, xinit, fargs)
    A, B, C, E = rendahl_coeffs(F, xss, fargs)
    F, Q = solve_system(A, B, C, E)
    irf = compute_irfs(F, Q, xss, irf_timeperiods)

    out = solution(F, Q, irf, xss, A, B, C, E)

    return out
    
end 
