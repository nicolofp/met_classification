using DataFrames, Arrow, Random, CSV, CategoricalArrays
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using Distances, StatsBase, LowRankModels, Clustering, Optim
using Plots, StatsPlots 

df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/GSE154829/GSE154829_cov.arrow"));
mRNA = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                             "/GSE154829/GSE154829_miRNA.arrow"));

size(mRNA)
Mrna = Matrix(Matrix{Float64}(mRNA[:,2:401])')

svd_mrna = svd(Mrna)
var_expl = (svd_mrna.S).^2 ./ sum((svd_mrna.S).^2)

# First 10 component explain ~98% of the variance
sum(var_expl[1:10])

scatter(svd_mrna.U[:,1],
        svd_mrna.U[:,2])

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #

Mrna_c = rescale(Mrna)                        

mv = DataFrame(media_ge = vec(mean(Mrna, dims = 1)),
               std_ge = vec(std(Mrna, dims = 1)))

scatter(mv[:,1],mv[:,2])
histogram(Mrna[:,6])
histogram(Mrna[:,249])
scatter(Mrna[:,1],df.bmi)

heatmap(cor(Mrna))

Dmat = hcat(string.(mRNA.Gene),zeros(2102,50))

# MultiThread --> automatically ordered  
Threads.@threads for i in 1:size(Mrna,2)
    println(i)
    m = minimum(Mrna[:,i])
    M = maximum(Mrna[:,i])
    p = (M-m)/50
    h = StatsBase.fit(Histogram, Mrna[:,i], collect(m:p:M))
    Dmat[i,2:51] = h.weights'
end 

asthma = df.asthma[1:400]
valid = [!ismissing(asthma[i]) for i in 1:400]
y = Array{Int64}(asthma[valid])
y = recode(y, 0=>0, 1:2=>1)
x = Mrna_c[valid,:]

include("ard_logistic_regression.jl")

# Sample using HMC.
m = logistic_ard(x, y)
chain_lard = sample(m, NUTS(), 1500);
quantile(chain_lard[:,1,1],0.025)

ard_res = hcat(string.(names(chain_lard)[1:4206]),zeros(4206,3))
for i in 1:4206
    ard_res[i,2] = quantile(chain_lard[:,i,1],0.025)
    ard_res[i,3] = mean(chain_lard[:,i,1])
    ard_res[i,4] = quantile(chain_lard[:,i,1],0.975)
end

ard_res = hcat(ard_res,sign.(ard_res[:,2]) .== sign.(ard_res[:,4]))

sum(ard_res[contains.(ard_res[:,1],Ref("β₁")),5])

histogram(Mrna[:,2])
histogram(rand(Poisson(mean(Mrna[:,2])),400),
          color = "orange")

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

Mrna_t = Matrix{Float64}(Mrna')
m = gamma_poisson(Mrna_t)
chain_gp = sample(m, NUTS(), 1500);


