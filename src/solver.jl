using Parameters
using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots
using QuantEcon
using LinearSolve

struct solution
    resultmessage::String
    F::Matrix{Float64}
    Q::Any
    xss::Vector{Float64}

    equations

    A::Matrix{Float64} 
    B::Matrix{Float64}
    C::Matrix{Float64}
    E::Any
end

# useful helper functions
atleast_2d(a) = fill(a,1,1)
atleast_2d(a::AbstractArray) = ndims(a) == 1 ? reshape(a, :, 1) : a

isinvertible(A) = !isapprox(det(A), 0, atol=1e-10)


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


function linear_coeffs(equations, xss, shocks_sd, fargs)

    shocks_sd = atleast_2d(shocks_sd)
    shocks_ss = zero(shocks_sd)

    A = ForwardDiff.jacobian(t -> equations(t, xss, xss, shocks_ss, shocks_sd, fargs...), xss)
    B = ForwardDiff.jacobian(t -> equations(xss, t, xss, shocks_ss, shocks_sd, fargs...), xss)
    C = ForwardDiff.jacobian(t -> equations(xss, xss, t, shocks_ss, shocks_sd, fargs...), xss)
    E = ForwardDiff.jacobian(t -> equations(xss, xss, xss, t, shocks_sd, fargs...), shocks_ss)

    return A, B, C, E

end


function solve_system(A, B, C, E, maxiter=1000, tol=1e-6; sing_solv=false)
    
    #==
    Solves for P and Q using Rehndal's Algorithm
    ==#
    
    F0 = zero(A)
    S0 = zero(A)
    error = one(tol) + tol
    iter = 0
    
    if sing_solv
        # pretty memory inefficient!
        μ=0.01
        M = I*μ
        Chat = C
        Bhat = B + 2.0*Chat*M
        Ahat = C*M^2 + B*M + A
        A = Ahat
        B = Bhat
        C = Chat
    end


    while error > tol && iter <= maxiter
        
        # using Linear solve for speed
        # probF = LinearProblem(-(C * F0 + B), A)
        # F1 = LinearSolve.solve(probF)
        F1 = -(C * F0 + B) \ A

        # probS = LinearProblem(-(C * F0 + B), A)
        # S1 = LinearSolve.solve(probS)

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
    
    if sing_solv
        F0 = F0 + I
        Q = -(C * F0 + B) \ E
    else
        Q = -(C * F0 + B) \ E
    end

    println(outmessage)

    return F0, Q, outmessage

    
end




function solve(equations, fargs, xinit; shocks_sd = 1.0)

    xss = get_ss(equations, xinit, shocks_sd, fargs)
    A, B, C, E = linear_coeffs(equations, xss, shocks_sd, fargs)

    if isinvertible(A) && isinvertible(C)
        F0, Q, outmessage = solve_system(A, B, C, E)
    else
        println("Either the A or C matrix isnt invertible, trying method with singular solvents")
        # else try method for singular solvent
        F0, Q, outmessage = solve_system(A, B, C, E; sing_solv=true)
    end

    out = solution(outmessage, F0, Q, xss, equations, A, B, C, E)

    return out
    
end 

function simdata(sol::solution, timeperiods::Int64)

    F = sol.F
    Q = sol.Q
    G = Matrix(I, size(F, 1), size(F, 1))

    lss = LSS(F, Q, G)

    X_simul, _ = simulate(lss, timeperiods)

    return X_simul

end


function compute_irfs(sol::solution, timeperiods = 40, shocks = nothing)

    IRF = zeros(size(sol.F, 1), timeperiods)

    if ~isnothing(shocks)
        shocks = atleast_2d(shocks)
        shocks_ss = zero(shocks)
        E = ForwardDiff.jacobian(t -> equations(sol.xss, sol.xss, sol.xss, t, shocks, fargs...), shocks_ss)
        Q = -(sol.C * sol.F + sol.B) \ E
    else
        Q = sol.Q
    end

    IRF[:, 1] = Q
    for t in 2:timeperiods
        IRF[:, t] = sol.F * IRF[:, t-1]
    end

    IRF = IRF .+ sol.xss

    return IRF

end


