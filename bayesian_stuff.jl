using LinearAlgebra, Distributions, Statistics, StatsBase
using Turing, Distributions, Optim, Zygote, ReverseDiff
using Plots, StatsPlots, LazyArrays, Memoization
using StatsFuns: logistic
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    p = p > 0 ? p : 1e-4 # numerical stability
    r = ϕ

    return NegativeBinomial(r, p)
end

# Analysis negative binomial 
X = rand(Normal(0.0,0.8),200,5)
betas = vec([0.3 -0.5 1.2 0.5 -1.1])

m_sim = exp.(1.5 .+ X * betas)
phi_sim = 0.82

y = [rand(NegativeBinomial2(m_sim[i],phi_sim)) for i in 1:200]

@model function nb_regression_ard(x,y) 
    
    # Size of the dataset
    N, D = size(x)

    # Set variance prior and ARD weights
    # λ ~ InverseGamma(1.0, 1.0)
    λ ~ Gamma(2.0, 2.0)
    α  ~ filldist(Gamma(2.0,2.0),D)

    # Set betas prior.
    β₀ ~ Normal(0, 20)
    β₁ ~ MvNormal(zeros(D), 1.0 ./ sqrt.(α))

    # Mean model
    z = β₀ .+ x * β₁
    mu = exp.(z)

    # Calculate the model
    for i in 1:N
        y[i] ~ NegativeBinomial2(mu[i],λ)
    end
end

m = nb_regression_ard(X,y)
chain_nb = sample(m, NUTS(), 10000);

@model function NegativeBinomialRegression(X, y)
    p = size(X, 2)
    n = size(X, 1)

    #priors
    λ ~ InverseGamma(0.1, 0.1)
    α ~ Normal(0, 10)
    β ~ filldist(Normal(0, 10), p)

    ## link
    z = α .+ X * β
    #z = X * β
    mu = exp.(z)

    #likelihood
    for i = 1:n
        y[i] ~ NegativeBinomial2(mu[i], λ)
    end
end

m1 = NegativeBinomialRegression(X,y)
chain_nb1 = sample(m1, NUTS(), 10000);

vcat(mean(Array(chain_nb[:,8:12,1]),dims = 1),
     mean(Array(chain_nb1[:,8:12,1]),dims = 1), 
     betas')

@model function negbinreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 10)
    # β ~ filldist(TDist(3), predictors)
    β ~ filldist(Normal(0, 10), predictors)
    # ϕ⁻ ~ Gamma(0.1, 0.1)
    # ϕ = 1 / ϕ⁻
    ϕ ~ InverseGamma(0.1, 0.1)

    #likelihood
    return y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), ϕ)))
end

m1 = NegativeBinomialRegression(X,y)
m2 = negbinreg(X,y)
chain_nb1 = sample(m1, NUTS(), 5000);
chain_nb2 = sample(m2, NUTS(), 5000);

a = rand(DirichletMultinomial(5, [0.5,0.25,0.25]))
a/sum(a)
rand(Dirichlet(10, 1))

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
    alpha ~ filldist(Exponential(1.0), nfeatures)
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

# Mixture Test
XM = rand(Normal(0.0,0.2),200,5)
XC = rand(Normal(0.0,0.2),200,2)
w = vec([0.5 0.2 0.2 0.05 0.05])
a = 1.1
b = -0.23
d = vec([1.1 -0.12])

for i in 1:5
    XM[:,i] = ecdf(XM[:,i])(XM[:,i])*4
end

mmix = a .+ b * (XM * w) .+ XC * d
y = rand(MvNormal(mmix,0.85 * I))

url = "https://envbran.s3.us-east-2.amazonaws.com/bwqs_tmp.csv"
DT = CSV.read(HTTP.get(url).body, DataFrame)
describe(DT)

XM = Matrix(DT[:,2:6])
XC = Matrix(DT[:,7:9])
y = DT[:,:y]

mx = bwqs_soft(XC, XM, y)
mx1 = bwqs_new(XC, XM, y)
mx2 = bwqs(XC, XM, y)
chain_mx = sample(mx, NUTS(), 1000);#, thinning=10, discard_initial=2000);
chain_mx1 = sample(mx1, NUTS(), 1000);
chain_mx2 = sample(mx2, NUTS(), 5000);
chain_mx2b = sample(mx2, NUTS(), MCMCThreads(), 5000, 1);


dump(chain_mx2)

# STD bwqs to account outlier and heavy tails
# Spatial models
# Mix model --> computational stabilty

