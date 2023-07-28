
using DataFrames, Arrow, Random, CSV, CategoricalArrays
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using Distances, StatsBase, LowRankModels
using Plots, StatsPlots 

df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/GSE154829/GSE154829_cov.arrow"));
mRNA = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                             "/GSE154829/GSE154829_miRNA.arrow"));

size(mRNA)
Mrna = Matrix(Matrix{Float64}(mRNA[:,2:401])')

svd_mrna = svd(Mrna)
var_expl = (svd_mrna.S).^2 ./ sum((svd_mrna.S).^2)

sum(var_expl[1:10])

scatter(svd_mrna.U[:,1],
        svd_mrna.U[:,2])

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #

Mrna_c = rescale(Mrna)                        

qpca_df = qpca(Mrna_c,2)
X,Y,ch = fit!(qpca_df)
scatter(qpca_df.X[1,:],qpca_df.X[2,:])

poiss_loss = GLRM(Mrna,PoissonLoss(),ZeroReg(),ZeroReg(),2)
X,Y,ch = fit!(poiss_loss)
scatter(poiss_loss.X[1,:],poiss_loss.X[2,:])

mv = DataFrame(media_ge = vec(mean(Mrna, dims = 1)),
               std_ge = vec(std(Mrna, dims = 1)))

scatter(mv[:,1],mv[:,2])
histogram(Mrna[:,6])

cutting = cut(Mrna[:,1],collect(m:p:M))

sort(countmap())

histogram(Mrna[:,249])
scatter(Mrna[:,1],df.bmi)

heatmap(cor(Mrna))

Dmat = hcat(string.(mRNA.Gene),zeros(2102,50))
#Threads.@threads 
for i in 1:size(Mrna,2)
    println(i)
    m = minimum(Mrna[:,i])
    M = maximum(Mrna[:,i])
    p = (M-m)/50
    h = StatsBase.fit(Histogram, Mrna[:,i], collect(m:p:M))
    Dmat[i,2:51] = h.weights'
end 

Dmat = Matrix{Float64}(Dmat[:,2:51])
svd_dmat = svd(Dmat)
var_dmat = (svd_dmat.S).^2 ./ sum((svd_dmat.S).^2)

sum(var_expl[1:10])

scatter(svd_mrna.U[:,1],
        svd_mrna.U[:,2])
