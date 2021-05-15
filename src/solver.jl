using Parameters
using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots

struct solution
    P::Matrix{Float}
    Q::Matrix{Float} 
    irf::Matrix{Float} 
    xss::Vector{Float}

    A::Matrix{Float} 
    B::Matrix{Float}
    C::Matrix{Float}
    E::Matrix{Float}
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
    
    P0 = zero(A)
    S0 = zero(A)
    error = one(tol) + tol
    iter = 0
    
    while error > tol && iter <= maxiter
        
        P1 = -(A * P0 + B) \ C
        S1 = -(C * S0 + B) \ A
        
        error = maximum(A * P1 * P1  + B * P1 + C)
        
        P0 .= P1
        S0 .= S1
        
        iter += 1
        
    end
    
    XP = LinearAlgebra.eigen(P0)
    XS = LinearAlgebra.eigen(S0)
    
    Q = -(A * P0 + B) \ E
    
    if iter == maxiter
        println("Convergence Failed. Max Iterations Reached. Error: $error")
    elseif maximum(abs.(XP.values)) > 1.0
        println("Non existence")
    elseif maximum(abs.(XS.values)) > 1.0
        println("No stable equilibrium")
    else
        println("Convergence Successful!")
    end
    
    return P0, Q

    
end


function compute_irfs(P, Q, xss, timeperiods = 40)

    IRF = zeros(size(P, 1), timeperiods)

    IRF[:, 1] = Q
    for t in 2:timeperiods
        IRF[:, t] = P * IRF[:, t-1]
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
    P, Q = solve_system(A, B, C, E)
    irf = compute_irfs(P, Q, xss, irf_timeperiods)

    out = solution(P, Q, irf, xss, A, B, C, E)

    return out
    
end 
