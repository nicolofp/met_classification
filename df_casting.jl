using DataFrames, Arrow

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
filter([:Methylation_Treatment, :Replicate] => (x,y) -> x == "OX" && 
         y == "rep1" , mapping)

#betas = permutedims(betas, 2)

