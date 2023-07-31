using DataFrames, Arrow, Random, CategoricalArrays
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using Distances, StatsBase, Clustering
using Plots, StatsPlots

# Rescale function
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ std(A, dims=dims) 

df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/GSE59491/Data/GSE59491_cov.arrow"));
genexpr = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                                "/GSE59491/Data/GSE59491_genexpr.arrow"));

# No missing data 
sum([sum(ismissing.(genexpr[:,i])) for i in 1:size(genexpr,2)])
sum([sum(ismissing.(df[:,i])) for i in 1:size(df,2)])

# Select first timepoint
df1 = df[df.sp_time .== 1,:]

# Svd --> no evident cluster
X_genexpr = Matrix(Matrix{Float64}(genexpr[:,string.(df1.geo_id)])')
X_genexpr = rescale(X_genexpr)
svd_genexpr = svd(X_genexpr)
scatter(svd_genexpr.U[:,1],svd_genexpr.U[:,2])
svd_genexpr.S.^2/sum(svd_genexpr.S.^2)

# Test mean difference 
T_genexpr = hcat(string.(genexpr.ID_REF),zeros(size(genexpr,1)))
Threads.@threads for i in 1:size(X_genexpr,2)
    T_genexpr[i,2] = pvalue(UnequalVarianceTTest(X_genexpr[df1.birth_out_1 .== 1,i], 
                                                 X_genexpr[df1.birth_out_1 .== 0,i]))
end
T_genexpr = hcat(T_genexpr,T_genexpr[:,2] .<= (0.05/size(X_genexpr,2)))
T_genexpr[T_genexpr[:,3] .== 1,:]

cor_genexpr = cor(X_genexpr)

Threads.@threads for i in 1:size(X_genexpr,2)
    X_tmp = Array{Float64, 2}(hcat(ones(size(X_genexpr,1)),
                              X_genexpr[:,i]))
    Y_tmp = Array{Float64}(df1.gestage)

    β = X_tmp\Y_tmp
    σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
    Σ = σ²*inv(X_tmp'*X_tmp)
    std_coeff = sqrt.(diag(Σ))
    
    RExp[i,2] = β[2]
    RExp[i,3] = std_coeff[2]
    RExp[i,4] = β[2]/std_coeff[2]
    RExp[i,5] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
    RExp[i,6] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
    RExp[i,7] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
end
ewas_results = DataFrame(RExp,:auto)
rename!(ewas_results, ["variable","beta","std_error","t_value","p_value","CI0025","CI0975"]);
ewas_results.p_value_bonf = (ewas_results.p_value .< 0.05/size(RExp)[1]);

genexpr.ID_REF .== "30850_at"
scatter(df1.gestage,X_genexpr[:,genexpr.ID_REF .== "30850_at"])

histogram(X_genexpr[:,1])
histogram!(rand(Poisson(mean(X_genexpr[:,1])),165))

