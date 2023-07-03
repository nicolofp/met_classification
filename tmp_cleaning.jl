using DataFrames, Arrow, Random
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using UMAP, Distances, StatsBase, TSne, Optim, MLBase, DecisionTree
using Impute: Substitute, impute
using Plots, StatsPlots 

df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/covariates.arrow"));
btpos = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/bupos.arrow"));
btpos_c = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/ST001828/btpos_code.arrow"));

# Extract names 
tmp = sum(ismissing.(Matrix(btpos[:,2:101])), dims = 2)
tmp1 = DataFrame(metabolite = Array(string.(btpos.metabolite)),
                 NAs = tmp[:,1]);

# Select the metabolites with <=25% of missing data
valid_met = tmp1[tmp1.NAs .<= 25,"metabolite"];
length(valid_met)

# Impute missing data with median 
M_imp = impute(Matrix(btpos[btpos.metabolite .∈ Ref(valid_met),2:101]), 
                            Substitute(; statistic=median); dims=:rows)

btpos_imp = DataFrame(hcat(valid_met, M_imp), :auto)
rename!(btpos_imp,Symbol.(vcat("metabolite",df.sample_ID))) 
       
M_btpos = Matrix(Matrix{Float64}(btpos_imp[:,2:101])')
#M_btpos = Matrix{Float64}(btpos_imp[:,2:101])
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #

M_btpos = rescale(M_btpos)
mean(M_btpos,dims = 1)
std(M_btpos,dims = 1)

svd_met = svd(M_btpos);
scatter(svd_met.U[:,1],svd_met.U[:,2])

valid_met
ttest_tmp = zeros(length(valid_met))
for i in 1:length(valid_met)
    ttest_tmp[i] = pvalue(UnequalVarianceTTest(M_btpos[df.sptb .== 1,i], 
                                               M_btpos[df.sptb .== 0,i]))
end
ttest_btpos = hcat(valid_met,ttest_tmp,ttest_tmp .<= (0.05/length(valid_met)))
sum(ttest_btpos[:,3] .== true)

# KS Test Perform an asymptotic two-sample Kolmogorov–Smirnov-test 
# of the null hypothesis that x and y are drawn from the same distribution 
# against the alternative hypothesis that they come from different distributions.
ks_pvalues = [pvalue(ApproximateTwoSampleKSTest(M_btpos[:,i],
                                                M_btpos[:,j])) for i in 1:length(valid_met), 
                                                                   j in 1:length(valid_met)] 
ks_group = [pvalue(ApproximateTwoSampleKSTest(M_btpos[df.sptb .== 1,i],
                                              M_btpos[df.sptb .== 0,i])) for i in 1:length(valid_met)]
valid_met[ks_group .< 0.05]
#btpos_c[btpos_c.Metabolite .∈ Ref(valid_met[ks_group .< 0.05]),:]

ks_pvalues[ks_group .< 0.05,ks_group .< 0.05]

# Random Forest classifier 
y = string.(df.sptb)
# x = M_btpos
x = M_btpos[:,btpos_imp.metabolite .∈ Ref(valid_met[ks_group .< 0.05])]
# train regression forest
# set of classification parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
#              multi-threaded forests must be seeded with an `Int`

model_rf = build_forest(y, x, -1, 1500, 0.7, -1, 3, 2, 0.0)
pred_rf = apply_forest(model_rf,x)
DecisionTree.confusion_matrix(pred_rf,y)

rf_importance = DataFrame(metabolite = valid_met[ks_group .< 0.05],
                          impurity = impurity_importance(model_rf), 
                          split = split_importance(model_rf))
sort!(rf_importance, [:impurity], rev = true)