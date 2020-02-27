using MLJ, Test, Pipe
using mm2020, CSVFiles, DataFrames

# Notes:
# winning model in 2019 used an xgboost model with a glmer measure of quality (RE's)
#   avg win rate in the last 14 days of the tournament
# Interesting features from second place:
#   Difference in the variance of game to game free throw percentage.
#   Difference in the variance of turnovers in the game to game free throw percentage.

# Get the submission sample
submission_sample = CSVFiles.load("/home/swojcik/github/mm2020.jl/data/MSampleSubmissionStage1_2020.csv") |> DataFrame

# Get the source seeds:
df_seeds = CSVFiles.load("/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MNCAATourneySeeds.csv") |> DataFrame

##############################################################
# Create training features for valid historical data
# SEEDS
seeds_features = make_seeds()
# EFFICIENCY
Wfdat, Lfdat, effdat = eff_stat_seasonal_means()
eff_features = get_eff_tourney_diffs(Wfdat, Lfdat, effdat)
# ELO
season_elos = elo_ranks(Elo())
elo_features = get_elo_tourney_diffs(season_elos)
### Full feature dataset
seeds_features_min = filter(row -> row[:Season] >= 2003, seeds_features)
eff_features_min = filter(row -> row[:Season] >= 2003, eff_features)
elo_features_min = filter(row -> row[:Season] >= 2003, elo_features)

# create full stub

stub = join(seeds_features_min, eff_features_min, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);
fdata = join(stub, elo_features, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);

exclude = [:Result, :Season, :LTeamID, :WTeamID]
deletecols!(fdata, exclude)

# Create features required to make submission predictions
seed_submission = get_seed_submission_diffs(submission_sample, df_seeds)
eff_submission = get_eff_submission_diffs(submission_sample, effdat) #see above
elo_submission = get_elo_submission_diffs(submission_sample, season_elos)
@test size(seed_submission, 1) == size(eff_submission, 1) == size(elo_submission, 1)

# Create full submission dataset
submission_features = hcat(seed_submission, eff_submission, elo_submission)

##########################################################################

# TRAINING

# Join the two feature sets
featurecols = [names(seed_submission), names(eff_submission), names(elo_submission)]
featurecols = collect(Iterators.flatten(featurecols))
fullX = [fdata[featurecols]; submission_features[featurecols]]
fullY = [seeds_features_min.Result; repeat([0], size(submission_features, 1))]

####################################0
using DataFrames, CSVFiles, Pipe, MLJ, LossFunctions
# reload the saved data
fullX = CSVFiles.load("data/fullX.csv") |> DataFrame
fullY = CSVFiles.load("data/fullY.csv") |> DataFrame

# create array of training and testing rows
train, test = partition(1:2230, 0.7, shuffle=true)
validate = [2230:size(fullY, 1)...]

# Recode result to win/ loss
y = @pipe categorical(fullY.y) |> recode(_, 0=>"lose",1=>"win");


#################################################
@load XGBoostClassifier()
xgb = XGBoostClassifier()
fullX_co = coerce(fullX, Count=>Continuous)
xgbm = machine(xgb, fullX_co, y)

# measuring the number of rounds
r = range(xgb, :num_round, lower=1, upper=50)
curve = learning_curve!(xgbm, resampling=CV(nfolds=3),
                        range=r, resolution=20,
                        measure=cross_entropy)

plot(curve.parameter_values, curve.measurements)

#--- Setting the rounds of the xgb, then tuning depth and children
xgb.num_round = 6
xgb.max_depth = 3
xgb.min_child_weight = 4.2105263157894735
xgb.gamma = 11
xgb.eta = .35
xgb.subsample = 0.6142857142857143
xgb.colsample_bytree = 1.0
r1 = range(xgb, :max_depth, lower = 3, upper = 10)
r2 = range(xgb, :min_child_weight, lower=0, upper=5)
tm = TunedModel(model = xgb, tuning = Grid(resolution = 20),
        resampling = CV(rng=11), ranges=[r1, r2],
        measure = cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows = train)

yhat = predict(mtm, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(mtm, rows=test), y[test])

###################################################
# TUNING GAMMA
xgbm = machine(xgb, fullX_co, y)
r = range(xgb, :gamma, lower=6, upper=20)
curve = learning_curve!(xgbm, resampling=CV(),
                        range=r, resolution=50,
                        measure=cross_entropy);
plot(curve.parameter_values, curve.measurements)
####################################################
# TUNING ETA
r = range(xgb, :eta, lower=.01, upper=.4)
tm = TunedModel(model = xgb, tuning = Grid(resolution = 20),
        resampling = CV(rng=11), ranges=r,
        measure = cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows = train)
######################################
# Tuning subsample and colsample
r1 = range(xgb, :subsample, lower=0.1, upper=1.0)
r2 = range(xgb, :colsample_bytree, lower=0.1, upper=1.0)
tm = TunedModel(model=xgb, tuning=Grid(resolution=8),
                resampling=CV(rng=234), ranges=[r1,r2],
                measure=cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows=train)

#########################################
# Tuning lam and colsample
r1 = range(xgb, :subsample, lower=0.1, upper=1.0)
r2 = range(xgb, :colsample_bytree, lower=0.1, upper=1.0)
tm = TunedModel(model=xgb, tuning=Grid(resolution=8),
                resampling=CV(rng=234), ranges=[r1,r2],
                measure=cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows=train)


#######################################
rf = @load RandomForestClassifier pkg="ScikitLearn"

rf_forest = EnsembleModel(atom=rf, n=1);
rf_model = machine(rf_forest, fullX, y)
fit!(rf_model, rows = train)
yhat = predict(rf_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(rf_model, rows=test), y[test])


####################################################

ada = @load AdaBoostStumpClassifier pkg="DecisionTree"
ada_model = machine(ada, fullX, y)
fit!(ada_model, rows = train)
yhat = predict(ada_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(ada_model, rows=test), y[test])

# Train the model!

# make the submission prediction
final_prediction = predict_mode(tree, rows=validate)


xg = @load GradientBoostingClassifier pkg = "ScikitLearn"
fullX_co = coerce(fullX, Count=>Continuous)
xg_model = machine(xg, fullX_co, y)
fit!(xg_model, rows = train)
yhat = predict(xg_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(xg_model, rows=test), y[test])

@load XGBoostClassifier()
xgb = XGBoostClassifier()
xg_model = machine(xgb, fullX_co, y)
fit!(xg_model, rows = train)
yhat = predict(xg_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(xg_model, rows=test), y[test])

#########################################################
# This is a working single model for XGBOOST Classifier
xgb_forest = EnsembleModel(atom=xgb, n=1000);
xg_model = machine(xgb_forest, fullX_co, y)
fit!(xg_model, rows = train)
yhat = predict(xg_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(xg_model, rows=test), y[test])
#######################################################3

# get parameter values and misclassification scores
miss_rates = ens_model.report.plotting.measurements[:, 1]
alphas = ens_model.report.plotting.parameter_values[:, 1]
################################


#####################################3

yhat = predict(ens_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict(ens_model, rows=test), y[test])

atom = @load RidgeRegressor pkg=MultivariateStats
mach = machine(ensemble, X, y)
########################################
################ WORKING EXAMPLE
tree_model = @load DecisionTreeClassifier verbosity=1
tree = machine(tree_model, fullX[:, [:SeedDiff]], y)
fit!(tree, rows = train)
yhat = predict(tree, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(tree, rows=test), y[test])

# Just checking on a GLM
df = hcat(fullY, fullX)
myform = @formula(y ~ SeedDiff + Diff_Pts_mean_mean)
mod = glm(myform, df[train, :], Binomial(), ProbitLink())
