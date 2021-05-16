# Rendahl Solver


The package uses the [Rendahl (2017)](https://www.ihs.ac.at/publications/eco/es-330.pdf) method for solving DSGE models, although the method can be easily applied to solving other models like heterogenous agent models ala Aiyagari.

Package was built because it seemed like there was no quick and dirty way of playing around with big macro models.

Using the package is simple. All you need to do is define a function `F(Xl, X, Xf, ϵ, args)` which outputs the residuals of the equiblibrium equations of the model. `Xf` are supposed to be the one period forward variables, `X` the current variables, and `Xl` the lagged variables. `ϵ` should contain all the shocks of the model. `args` are optional arguments you may want to pass in like parameters the model needs.


# An Example - A simple RBC Model

## Setting Up
For example the `F` function for an RBC model might look like:

```julia
function F(Xl, X, Xf, ϵ, params)
    #====================
    Xf = X_{t+1}
    X = X_t
    Xl = X_{t-1}
    ϵ = ϵ_t
    ======================#
    Cf, Rf, Kf, Yf, Zf = Xf
    C, R, K, Y, Z = X
    Cl, Rl, Kl, Yl, Zl = Xl
    
    @unpack β, α, γ, δ, ρ, σz = params
    
    ϵ = ϵ[1]
    
    # RBC Equations
    residual = [1.0 - β * Rf * Cf^(-γ) * C^(γ);
                R - α*Z*Kl^(α-1) - 1 + δ;
                K - (1-δ)*Kl - Y + C;
                Y - Z*Kl^α;
                log(Z) - ρ*log(Zl) - σz*ϵ]
    
    return residual
    
end
```


The `params` object here is a named tuple constructed with the help of the [Parameters.jl](https://github.com/mauro3/Parameters.jl) package so that interfacing with it is easy. For reference here is how it is constructed


```julia
rbc = @with_kw (
    β = 0.96,
    α = 0.33,
    γ = 2.0,
    δ = 0.1,
    ρ = 0.9,
    σz = 0.8 # shock for epsilon
)

params = rbc()
```

But how you wish to put in the parameters is totally up to you!

## Solving

Solving the model is straightforward. We use the `solve` function to get out a `Solution` object with the fields `(P, Q, irf, :xss, A, B, C, E)`. 

```julia
sol = solve(F, [params], 
            xinit = ones(5), 
            irf_timeperiods = 40)
```

And that's it! If you want to plot out the irfs, `draw_irf` is a nice helper function.


```julia
draw_irf(sol.irf, sol.xss, ["Consumption", "Interest Rate", "Capital", "Output", "Z"])
```

The output will look like this:

![]("src/rbc_irf.png")


