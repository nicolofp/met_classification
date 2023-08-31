# Load Distributed to add processes and the @everywhere macro.
using Distributed

# Load Turing.
using Turing

# Add four processes to use for sampling.
addprocs(4; exeflags="--project=$(Base.active_project())")

# Initialize everything on all the processes.
# Note: Make sure to do this after you've already loaded Turing,
#       so each process does not have to precompile.
#       Parallel sampling may fail silently if you do not do this.
@everywhere using Turing
@everywhere using DataFrames
@everywhere using CSV

# Define a model on all processes.
@everywhere @model function bwqs_new(cx, mx, y)
    
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

@everywhere url = "C:/Users/nicol/Documents/bwqs_tmp.csv"
@everywhere DT = CSV.read(url, DataFrame)

@everywhere XM = Matrix(DT[:,2:6])
@everywhere XC = Matrix(DT[:,7:9])
@everywhere y = DT[:,:y]

# Declare the model instance everywhere.
@everywhere model_bwqs_multi = bwqs_new(XC, XM, y)
model_bwqs = bwqs_new(XC, XM, y)

# Sample four chains using multiple processes, each with 1000 samples.
chain_multi = sample(model_bwqs_multi, NUTS(), MCMCDistributed(), 50000, 1)
chain_simpl = sample(model_bwqs, NUTS(), 50000)