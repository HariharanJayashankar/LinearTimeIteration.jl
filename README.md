# A DSGE Solver using Rendahl (2017)


The package uses the Linear Time Iteration method from [Rendahl (2017)](https://www.ihs.ac.at/publications/eco/es-330.pdf) for solving linear rational expectation models like those typically found in DSGE models.

To install simply open your Julia terminal, go to the package manager by inputting `]` and then copying in:

```
add https://github.com/HariharanJayashankar/Rendahl.jl
```

Package was built because it seemed like there was no quick way of playing around with big macro models.

Using the package is simple. All you need to do is define a function `equations(Xl, X, Xf, ϵ, ϵ_sd ...)` which outputs the residuals of the equiblibrium equations of the model. `Xf` are supposed to be the one period forward variables, `X` the current variables, and `Xl` the lagged variables. `ϵ` should contain all the shocks of the model.

`ϵ_sd` contains the standard deviations to all the shocks. This value is a bit important as it determines what the final solution will look like.

`args` are optional arguments you may want to pass in like parameters the model needs.


# An Example - A simple RBC Model

## Setting Up
For example the `equations` function for an RBC model might look like:

```julia
function equations(Xl, X, Xf, ϵ, ϵ_sd, params)

    Cf, Rf, Kf, Yf, Zf = Xf
    C, R, K, Y, Z = X
    Cl, Rl, Kl, Yl, Zl = Xl
    
    @unpack β, α, γ, δ, ρ = params
    
    ϵ = ϵ[1] * ϵ_sd[1]
    
    # RBC Equations
    residual = [1.0 - β * Rf * Cf^(-γ) * C^(γ);
                R - α*Z*Kl^(α-1) - 1 + δ;
                K - (1-δ)*Kl - Y + C;
                Y - Z*Kl^α;
                log(Z) - ρ*log(Zl) - ϵ]
    
    return residual
    
end
```

As you can see the function has an additional argument at the end called `parameters`. In this case it is the tuple of parameters which the model will read. It is constructed with the help of [Parameters.jl](https://github.com/mauro3/Parameters.jl). For reference here is how it is constructed in this case:


```julia
rbc = @with_kw (
    β = 0.96,
    α = 0.33,
    γ = 2.0,
    δ = 0.1,
    ρ = 0.9
)

params = rbc()
```

But how you wish to pass in the parameters is totally up to you!

## Solving

Solving the model is straightforward. We use the `solve` function to get out a `Solution` object with the fields `(resultmessage, F, Q, xss, equations, A, B, C, E)`. `F` and `Q` define the solution to the system. `xss` are the steady state values of the variables. `equations` stores the function which outputs the residual of the equilibrium equations. `A, B, C, E` correspond to their namesakes in the paper.

```julia
shocks_sd = 0.8
sol = solve(equations, [params], shocks_sd, xinit = ones(5))
```

`sol` has everything we need to understand our model.

If you want to simulate some data you can use the `simdata` function provided:

```julia
sim = simdata(sol, 200)
plot(sim', layout = 5, title = ["Consumption", "Interest Rate", "Capital", "Output", "Z"])
```

Similarly if you want to look at how the IRF's look:

```julia
irf = compute_irfs(sol, 100)
plot(irf', layout = 5, title = ["Consumption", "Interest Rate", "Capital", "Output", "Z"], labels = "")
```

The output will look like this:

![RBC IRF](https://user-images.githubusercontent.com/32820850/118377767-f8710d80-b5ec-11eb-9b54-4d96ba17f65b.png)

