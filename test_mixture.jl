using LinearAlgebra, Distributions, Statistics, StatsBase
using Turing, Distributions, Optim, Zygote, ReverseDiff, Memoization
using Plots, StatsPlots, LazyArrays, Random
using StatsFuns: logistic
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Mixture Test
rng = MersenneTwister(1990);
N = 342
XM = rand(rng, Normal(0.0,0.2),N,5)
XC = rand(rng, Normal(0.0,0.2),N,2)
w = vec([0.5 0.2 0.2 0.05 0.05])
a = 1.1
b = -0.23
d = vec([1.1 -0.12])

for i in 1:5
    XM[:,i] = ecdf(XM[:,i])(XM[:,i])*4
end

mmix = a .+ b * (XM * w) .+ XC * d
y = rand(rng,MvNormal(mmix,0.85 * I))



@model function bwqs_new(cx, mx, y)
    
    # Set variance prior.
    σ₂ ~ Gamma(2.0, 2.0)

    # Set intercept prior.
    intercept ~ Normal(0, 20)
    beta ~ Normal(0, 20)
    ncovariates = size(cx, 2)
    delta ~ filldist(Normal(0, 20), ncovariates)

    # Set the priors on our coefficients.
    nfeatures = size(mx, 2)
    alpha ~ filldist(Gamma(2.0, 2.0), nfeatures)
    w ~ Dirichlet(alpha) 

    # Calculate all the mu terms. 
    mu = intercept .+ beta * (mx * w) .+ cx * delta

    return y ~ MvNormal(mu, sqrt(σ₂))
end

softmax(x) = exp.(x) ./ sum(exp.(x))

function DirichletLogit(μ, ϕ)
    α = softmax(μ) * ϕ
    return Dirichlet(α)
end


@model function bwqs_soft(cx, mx, y)
    
    # Set variance prior.
    σ₂ ~ Gamma(2.0, 2.0)

    # Set intercept prior.
    intercept ~ Normal(0, 10)
    beta ~ Normal(0, 10)
    ncovariates = size(cx, 2)
    delta ~ filldist(Normal(0, 20), ncovariates)

    # Set the priors on our coefficients.
    nfeatures = size(mx, 2)
    alpha ~ filldist(Gamma(1.0,1.0), nfeatures)
    phi ~ Gamma(2.0,2.0)
    w ~ DirichletLogit(alpha, phi) 
    #w = softmax(alpha)

    # Calculate all the mu terms. 
    mu = intercept .+ beta * (mx * w) .+ cx * delta

    return y ~ MvNormal(mu, sqrt(σ₂))
end

@model function bwqs(cx, mx, y)
    # Set variance prior.
    σ₂ ~ Gamma(2.0, 2.0)

    # Set intercept prior.
    intercept ~ Normal(0, 20)
    beta ~ Normal(0, 20)
    ncovariates = size(cx, 2)
    delta ~ filldist(Normal(0, 20), ncovariates)

    # Set the priors on our coefficients.
    nfeatures = size(mx, 2)
    w ~ Dirichlet(nfeatures, 1) 

    # Calculate all the mu terms.
    mu = intercept .+ beta * (mx * w) .+ cx * delta
    return y ~ MvNormal(mu, sqrt(σ₂))
end


mx1 = bwqs_soft(XC, XM, y)
mx2 = bwqs_new(XC, XM, y)
mx3 = bwqs(XC, XM, y)
chain_mx1 = sample(mx1, NUTS(), 2000);#, thinning=10, discard_initial=2000);
chain_mx2 = sample(mx2, NUTS(), 1000);
chain_mx3 = sample(mx3, NUTS(), 1000);

describe(chain_mx1)[1]