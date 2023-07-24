using DataFrames, Arrow, Random, FillArrays
using LinearAlgebra, Distributions, Statistics, StatsBase
using Turing, Distributions, Optim, Zygote, ReverseDiff
using Impute: Substitute, impute
using Plots, StatsPlots 

Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) 
#std(A, dims=dims) #

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

M_buneg = rescale(M_buneg)
mean(M_buneg,dims = 1)
std(M_buneg,dims = 1)

@model function pPCA(X::AbstractMatrix{<:Real}, k::Int)
    # retrieve the dimension of input matrix X.
    N, D = size(X)

    # weights/loadings W
    W ~ filldist(Normal(), D, k)

    # latent variable z
    Z ~ filldist(Normal(), k, N)

    # mean offset
    μ ~ MvNormal(Eye(D))
    c_mean = W * Z .+ reshape(μ, D, 1)
    return X ~ arraydist([MvNormal(m, Eye(N)) for m in eachcol(c_mean')])
end;

ppca = pPCA(M_buneg,2)

# ADVI
advi = ADVI(10, 1000)
q = vi(ppca, advi);

chain_ppca = sample(ppca, NUTS(), 1000);

# Extract parameter estimates for predicting x - mean of posterior
W = reshape(mean(group(chain_ppca, :W))[:, 2], (5010, 2))
Z = reshape(mean(group(chain_ppca, :Z))[:, 2], (2, 100))
μ = mean(group(chain_ppca, :μ))[:, 2]

df_pca = DataFrame(Z', :auto)
rename!(df_pca, Symbol.(["z" * string(i) for i in collect(1:2)]))
#df_pca[!, :type] = repeat([1, 2]; inner=n_cells ÷ 2)

scatter(df_pca[:, :z1], 
        df_pca[:, :z2]; 
        xlabel="z1", ylabel="z2")#, group=df_pca[:, :type])


vi_res = mean(rand(q, 1000); dims=2)
tmp_vi = hcat(vi_res[10021:10220][collect(1:2:200)],
              vi_res[10021:10220][collect(2:2:200)])

tmp_vi = tmp_vi * Matrix([-1 0; 0 -1])

scatter!(tmp_vi[:, 1],   
        tmp_vi[:, 2]; 
        xlabel="z1", ylabel="z2", col="red")


@model function pPCA_dir(X::AbstractMatrix{<:Real}, k::Int)
        # retrieve the dimension of input matrix X.
        N, D = size(X)
        
        # weights/loadings W
        #W ~ filldist(Normal(), D, k)
        W ~ filldist(Dirichlet(D, 1), k)
        
        # latent variable z
        Z ~ filldist(Normal(), k, N)
        
        # mean offset
        μ ~ MvNormal(Eye(D))
        c_mean = W * Z .+ reshape(μ, D, 1)
        return X ~ arraydist([MvNormal(m, Eye(N)) for m in eachcol(c_mean')])
end;

ppca = pPCA_dir(M_buneg,1)

#rand(filldist(Normal(), 10, 1))
#sum(rand(filldist(Dirichlet(10, 1), 1)))

# ADVI
advi = ADVI(10, 1000)
q = vi(ppca, advi);

chain_ppca = sample(ppca, NUTS(), 1000)

chain_ppca[:,1:100,:]

W = reshape(mean(group(chain_ppca, :W))[:, 2], (10, 1))
Z = reshape(mean(group(chain_ppca, :Z))[:, 2], (1, 100))
μ = mean(group(chain_ppca, :μ))[:, 2]

sum(W)

mat_rec = W * Z .+ repeat(μ; inner=(1, 100))
M_buneg