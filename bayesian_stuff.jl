using LinearAlgebra, Distributions, Statistics, StatsBase
using Turing, Distributions, Optim, Zygote, ReverseDiff
using Plots, StatsPlots
using StatsFuns: logistic
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)


# Analysis negative binomial 
tmp = rand(NegativeBinomial(10,0.5),1000)
mean(tmp)
var(tmp)

p = mean(tmp)/var(tmp)
r = mean(tmp)^2/(var(tmp) - mean(tmp))

X = rand(Normal(0.0,0.02),200,5)
betas = vec([-0.3 -0.5 -0.2 1.1 0.2])

m_sim = 0.5 .+ X * betas
var_sim = 0.82

rs = (m_sim.^2)./(var_sim .- m_sim)
ps = m_sim./var_sim
y = [rand(NegativeBinomial(rs[i],ps[i])) for i in 1:200]

@model function nb_regression_ard(x,y) 
    
    # Size of the dataset
    N, D = size(x)

    # Set variance prior and ARD weights
    σ₂ ~ Gamma(2.0, 2.0)
    α  ~ filldist(Gamma(2.0,2.0),D)

    # Set betas prior.
    β₀ ~ Normal(0, 20)
    β₁ ~ MvNormal(zeros(D), 1.0 ./ sqrt.(α))

    # Calculate the model
    for i in 1:N
        m = β₀ .+ x[i, :]' * β₁
        v = σ₂
        p = logistic(m/v)
        r = exp((m^2)/(v-m))
        y[i] ~ NegativeBinomial(r,p)
    end
end

m = nb_regression_ard(X,y)
chain_nb = sample(m, NUTS(), 15000);
