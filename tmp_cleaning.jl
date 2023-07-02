using DataFrames, Arrow
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using UMAP, Distances, StatsBase, TSne, Optim
using Impute: Substitute, impute
using Plots, StatsPlots 

df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/covariates.arrow"));
btpos = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/btpos.arrow"));
btpos_c = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/btpos_code.arrow"));

# Extract names 
tmp = sum(ismissing.(Matrix(btpos[:,2:101])), dims = 2)
tmp1 = DataFrame(metabolite = Array(string.(btpos.Metabolite)),
                 NAs = tmp[:,1]);

# Select the metabolites with <=25% of missing data
valid_met = tmp1[tmp1.NAs .<= 25,"metabolite"];

# Impute missing data with median 
M_imp = impute(Matrix(btpos[btpos.Metabolite .∈ Ref(valid_met),2:101]), 
                            Substitute(; statistic=median); dims=:rows)

btpos_imp = DataFrame(hcat(valid_met, M_imp), :auto)
rename!(btpos_imp,Symbol.(vcat("metabolite",df.sample_ID))) 
       
#M_btpos = Matrix(Matrix{Float64}(btpos_imp[:,2:101])')
M_btpos = Matrix{Float64}(btpos_imp[:,2:101])
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #

M_btpos = rescale(M_btpos)
mean(M_btpos,dims = 1)
std(M_btpos,dims = 1)

svd_met = svd(M_btpos);
scatter(svd_met.U[:,1],svd_met.U[:,2])