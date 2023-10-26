using DataFrames, Arrow, Plots

df = DataFrame(Arrow.Table("G:/My Drive/Dati/MMIP/" * 
                            "epi.arrow"));
exposure = DataFrame(Arrow.Table("G:/My Drive/Dati/MMIP/" *
                                 "exposure.arrow"));
betas = DataFrame(Arrow.Table("G:/My Drive/Dati/MMIP/" *
                              "betas.arrow"));      
mapping = DataFrame(Arrow.Table("G:/My Drive/Dati/MMIP/" *
                                "mapping.arrow"));
codebook = DataFrame(Arrow.Table("G:/My Drive/Dati/MMIP/" *
                                 "codebook.arrow"));                   

# Short mapping --> all at birth
mapping = mapping[:,[:Sample_Name, :Methylation_Treatment, :Replicate, :CHILD_PID]]
tmp_mapping = filter([:Methylation_Treatment, :Replicate] => (x,y) -> x == "OX" && 
         y == "rep1" , mapping)

betas[:,vcat("rn",tmp_mapping.Sample_Name)]
histogram(Array(betas[1,tmp_mapping.Sample_Name]), 
          title = betas[1,"rn"],
          label = "", bins = 10)

exposure[:,[:CHILD_PID, :Chemical_Group, :Analyte_Code, 
            :Units, :LOD, :Concentration,:Comment_code]]

unique(exposure.Chemical_Group)

exp_red = filter([:CHILD_PID,:Comment_code, :Chemical_Group] => (x,y,z) -> x ∈ tmp_mapping.CHILD_PID 
                    #&& y == 0 
                    #&& z == "UTE"
                    , exposure)            

dropmissing!(exp_red)
dropmissing(unstack(exp_red, :CHILD_PID, :Analyte_Code, :Concentration))

exp_red.Concentration .<= 0
exp_red[exp_red.Concentration .< 0,:Concentration] .== 0 

exposure[exposure.Concentration .< 0,:]