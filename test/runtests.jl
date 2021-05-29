using rendahl
using Test
using Parameters

function rbc_eq(Xl, X, Xf, ϵ, params)
    #====================
    Xf = X_{t+1}
    X = X_t
    Xl = X_{t-1}
    ϵ = ϵ_t
    ======================#
    Cf, Rf, Kf, Yf, Zf = Xf
    C, R, K, Y, Z = X
    Cl, Rl, Kl, Yl, Zl = Xl
    
    @unpack β, α, γ, δ, ρ = params
    
    ϵ = ϵ[1]
    
    # RBC Equations
    residual = [1.0 - β * Rf * Cf^(-γ) * C^(γ);
                R - α*Z*Kl^(α-1) - 1 + δ;
                K - (1-δ)*Kl - Y + C;
                Y - Z*Kl^α;
                log(Z) - ρ*log(Zl) - ϵ]
    
    return residual
    
end

rbc = @with_kw (
    β = 0.96,
    α = 0.33,
    γ = 2.0,
    δ = 0.1,
    ρ = 0.9
)

params = rbc()



function rbc_ss(params)
    
    #===========
    Get the steady state values
    for X
    ============#
    
    @unpack β, α, γ, δ, ρ = params
    
    Z = 1
    R = 1/β
    K = ((R - 1 + δ)/α)^(1/(α - 1))
    Y = K^α
    C = Y - δ*K
    
    ss = [C, R, K, Y, Z]
    
    return ss
end

func_ss = rendahl.get_ss(rbc_eq, ones(5), [params])
closedform_rbc_ss = rbc_ss(params)

@testset "rendahl.jl" begin

    @test isapprox(func_ss, closedform_rbc_ss)
    
end
