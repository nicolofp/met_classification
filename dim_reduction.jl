using DataFrames, Arrow, Random
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using Distances, StatsBase, LowRankModels
using Impute: Substitute, impute
using Plots, StatsPlots 

df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/covariates.arrow"));
buneg = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/buneg.arrow"));
# Extract names 
tmp = sum(ismissing.(Matrix(buneg[:,2:101])), dims = 2)
tmp1 = DataFrame(metabolite = Array(string.(buneg.metabolite)),
                 NAs = tmp[:,1]);

# Select the metabolites with <=25% of missing data
valid_met = tmp1[tmp1.NAs .<= 25,"metabolite"];
length(valid_met)

# Impute missing data with median 
M_imp = impute(Matrix(buneg[buneg.metabolite .∈ Ref(valid_met),2:101]), 
                            Substitute(; statistic=median); dims=:rows)

buneg_imp = DataFrame(hcat(valid_met, M_imp), :auto)
rename!(buneg_imp,Symbol.(vcat("metabolite",df.sample_ID))) 
       
M_buneg = Matrix(Matrix{Float64}(buneg_imp[1:10,2:101])')
#M_buneg = Matrix{Float64}(buneg_imp[:,2:101])
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #

M_buneg = rescale(M_buneg)
mean(M_buneg,dims = 1)
std(M_buneg,dims = 1)                        

qpca_buneg = pca(M_buneg,2)
X,Y,ch = fit!(qpca_buneg)
scatter(qpca_buneg.X[1,:],qpca_buneg.X[2,:])

loss = QuadLoss()
rx = SimplexConstraint()
ry = ZeroReg()
simplex_pca = GLRM(M_buneg,loss,rx,ry,10)
X,Y,ch = fit!(simplex_pca)
sum(X[:,1])
Y
X
X'*Y  
M_buneg



