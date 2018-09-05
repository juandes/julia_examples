using RDatasets, DataFrames, GLM, Plots, StatPlots, DecisionTree
using ScikitLearn.CrossValidation: cross_val_score
# prepare the plots backend
gr(size=(400,300)) #plots backend

# load Iris dataset
iris = dataset("datasets", "iris")
# plot Iris
@df iris scatter(:SepalLength, :SepalWidth, group=:Species,
        title = "Iris Dataset",
        xlabel = "Length", ylabel = "Width",
        bg=RGB(.2,.2,.2))



# create a small dataframe
data = DataFrame(X=[1,2,3], Y=[2,4,7])
# Linear Regression trained via ordinary least squares
basic_lm = lm(@formula(Y ~ X), data)
print(basic_lm)
# load the mtcars dataset
df = dataset("datasets", "mtcars")
mt_lm = fit(LinearModel, @formula(MPG ~ Cyl), df)
print(basic_lm)

#Decision trees
features, labels = load_data("iris")
features = float.(features)
labels   = string.(labels)

model = DecisionTreeClassifier(max_depth=2)

DecisionTree.fit!(model, features, labels)
print_tree(model, 5)
DecisionTree.predict(model, [5.9,3.0,5.1,1.9])
# get the probability of each label
predict_proba(model, [5.9,3.0,5.1,1.9])
accuracy = cross_val_score(model, features, labels, cv=3)
accuracy
