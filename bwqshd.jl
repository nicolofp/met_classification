using LinearAlgebra, Distributions, Statistics, StatsBase
using Turing, Distributions, Optim, Zygote, ReverseDiff
using Plots, StatsPlots, LazyArrays
using StatsFuns: logistic
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

@model function hbwqs(
    M, X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2), nmix=size(M, 2)
    )
    #priors
    # α ~ Normal(mean(y), 2.5 * std(y))                     # population-level intercept
    σ ~ Exponential(std(y))                                 # residual SD
    #prior for variance of random intercepts and slopes
    #usually requires thoughtful specification
    τₐ ~ truncated(Cauchy(0, 2); lower=0)                 # group-level SDs intercepts
    δ ~ filldist(Normal(0, 20), predictors)               # Covariates fixed  
    αₖ ~ filldist(Gamma(2.0, 2.0), nmix)              # Prior on Dirichlet alphas
    w ~ Dirichlet(αₖ)                                      # Dirichlet on simplex
    #τᵦ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr)  # group-level slopes SDs
    τᵦ ~ truncated(Cauchy(0, 2); lower=0)
    αⱼ ~ filldist(Normal(0, τₐ), n_gr)                     # group-level intercepts
    βⱼ ~ filldist(Normal(0, τᵦ), n_gr)          # group-level standard normal slopes

    #likelihood
    # ŷ = α .+ αⱼ[idx] .+ X * βⱼ * τᵦ
    # ŷ = αⱼ[idx] .+ (M * w) * βⱼ * τᵦ .+ X * δ
    # return y ~ MvNormal(ŷ, σ^2 * I)
    for i in 1:size(X,1)
        mu = αⱼ[idx[i]] + (M[i,:]' * w) * βⱼ[idx[i]] + X[i,:]' * δ
        y[i] ~ Normal(mu,σ^2)
    end
end;

# Example from: https://storopoli.io/Bayesian-Julia/pages/11_multilevel_models/

using DataFrames
using CSV
using HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/cheese.csv"
cheese = CSV.read(HTTP.get(url).body, DataFrame)
describe(cheese)

# cheese:     type of cheese from A to D
# rater:      id of the rater from 1 to 10
# background: type of rater, either rural or urban
# y:          rating of the cheese

for c in unique(cheese[:, :cheese])
    cheese[:, "cheese_$c"] = ifelse.(cheese[:, :cheese] .== c, 1, 0)
end

cheese[:, :background_int] = map(cheese[:, :background]) do b
    if b == "rural"
        1
    elseif b == "urban"
        2
    else
        missing
    end
end

first(cheese, 5)

X = Matrix(select(cheese, Between(:cheese_A, :cheese_D)));
y = cheese[:, :y];
idx = cheese[:, :background_int];

@model function hbwqs2(
    M, X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2), nmix=size(M, 2)
    )
    #priors
    # α ~ Normal(mean(y), 2.5 * std(y))                     # population-level intercept
    σ ~ Exponential(std(y))                                 # residual SD
    #prior for variance of random intercepts and slopes
    #usually requires thoughtful specification
    τₐ ~ truncated(Cauchy(0, 2); lower=0)                 # group-level SDs intercepts
    δ ~ filldist(Normal(0, 20), predictors)               # Covariates fixed  
    αₖ ~ filldist(Gamma(2.0, 2.0), nmix)              # Prior on Dirichlet alphas
    w ~ Dirichlet(αₖ)                                      # Dirichlet on simplex
    τᵦ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr)  # group-level slopes SDs
    #τᵦ ~ truncated(Cauchy(0, 2); lower=0)
    αⱼ ~ filldist(Normal(0, τₐ), n_gr)                     # group-level intercepts
    βⱼ ~ filldist(Normal(0, 1), 1, n_gr)          # group-level standard normal slopes

    #likelihood
    # ŷ = α .+ αⱼ[idx] .+ X * βⱼ * τᵦ
    ŷ = αⱼ[idx] .+ (M * w) * βⱼ * τᵦ .+ X * δ
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

url = "C:/Users/nicol/Documents/bwqs_tmp.csv"
DT = CSV.read(url, DataFrame)
describe(DT)

M = Matrix(select(DT, Between(:UAs_q, :UPb_q))); 
X = Matrix(DT[:,vcat(8:12,15:17)]);
y = DT[:, :y];
idx = DT[:, :cohort];

model_intercept = hbwqs2(M, X, idx, y)
chain_intercept = sample(model_intercept, NUTS(), 1000)#, MCMCThreads(), 1_000, 4)

tmp = [1, 3]
tmp[idx]

