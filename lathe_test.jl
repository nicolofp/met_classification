# https://github.com/emmaccode/Lathe-Books/tree/main/models
# Example with lathe
using CSV
using DataFrames 
using Lathe
using Lathe.preprocess: TrainTestSplit
using Lathe.models: RandomForestClassifier
using Lathe.stats: catacc
df = CSV.read("C:/Users/nicol/Documents/bwqs_tmp.csv", DataFrame)

for name in names(df)
    println(name)
end

df = dropmissing(df)
train, test = TrainTestSplit(df)

target = :y
feature = :m_age
trainX = train[!, feature]
trainy = train[!, target]
testX = test[!, feature]
testy = test[!, target]

model = RandomForestClassifier(trainX, trainy)
y_hat = model.predict(testX)
