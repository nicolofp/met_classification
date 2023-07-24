using DataFrames, Arrow, Random
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using Distances, StatsBase, LowRankModels
using Impute: Substitute, impute
using Plots, StatsPlots 

# "C:\Users\nicol\Documents\Datasets\GSE108497\GSE108497_miRNA_norm.arrow"
df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/GSE108497/GSE108497_miRNA_norm.arrow"));
size(df)
first(df)
# Extract names 
tmp = sum(ismissing.(Matrix(df[:,2:513])), dims = 2)
tmp1 = DataFrame(ID_REF = Array(string.(df.ID_REF)),
                 NAs = tmp[:,1]);

# Select the metabolites with <=25% of missing data
valid_met = tmp1[tmp1.NAs .<= 0,"ID_REF"];
length(valid_met)

tmp2 = df.ID_REF .∈ Ref(valid_met)
df = df[tmp2,:]
       
M_df = Matrix(Matrix{Float64}(df[:,2:513])')
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #

M_df = rescale(M_df)                        

qpca_df = qpca(M_df,2)
X,Y,ch = fit!(qpca_df)
scatter(qpca_df.X[1,:],qpca_df.X[2,:])







