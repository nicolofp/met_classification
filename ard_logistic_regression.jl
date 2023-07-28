using Turing, Distributions, Optim, Zygote, ReverseDiff
using StatsFuns: logistic
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Bayesian logistic regression with ARD (Automatic Relevance Determination).
@model function logistic_ard(x, y) 
    
    # Size of the dataset
    N, D = size(x)

    # Set variance prior and ARD weights
    σ₂ ~ Gamma(2.0, 2.0)
    α  ~ filldist(Gamma(2.0,2.0),D)

    # Set betas prior.
    β₀ ~ Normal(0, 20)
    β₁ ~ MvNormal(zeros(D), 1.0 ./ sqrt.(α))
    #β₁ ~ MvNormal(zeros(D), ones(D))

    # Calculate the model
    for i in 1:N
        v = logistic(β₀ .+ x[i, :]' * β₁)
        y[i] ~ Bernoulli(v)
    end
end


