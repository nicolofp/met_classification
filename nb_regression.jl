
using DataFrames, Arrow, Random, CSV
using LinearAlgebra, Distributions, Statistics, HypothesisTests
using Distances, StatsBase, LowRankModels
using Plots, StatsPlots 

df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                            "/GSE154829/GSE154829_cov.arrow"));
mRNA = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets" * 
                             "/GSE154829/GSE154829_miRNA.arrow"));

