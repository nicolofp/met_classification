using LinearAlgebra, Distributions, Statistics, StatsBase
using Turing, Distributions, Optim, Zygote, ReverseDiff
using Plots, StatsPlots, Random, DelimitedFiles
using StatsFuns: logistic
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# generate data
Random.seed!(1234)
back_distr = rand(Gamma(5.0,2.0),100)
X = zeros(100,40)
for i in 1:100
    X[i,:] = rand(Poisson(back_distr[i]),40)
end

@model function gamma_poisson(x)
    # Size of the sample
    N, D = size(x)
    
    # Prior of the Gamma background
    a0 ~ Gamma(1,1)
    b0 ~ Gamma(1,1)
    
    # background rate
    br ~ filldist(Gamma(a0,b0),N)

    for n in 1:N
        x[n,:] ~ Poisson(br[n])
    end
end;

# Instantiate model
m = gamma_poisson(X)
chain_gp = sample(m, NUTS(), 1000);

#writedlm("gamma_poisson.log", Array(chain_gp[:,:,1]))
writedlm("gamma_poisson_1.log", summarystats(chain_gp))

tmp = summarystats(chain_gp)
open("met_classification/gamma_poisson_1.txt", "w") do file
    write(file, tmp)
end

dump(chain_nb1)


