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

# create array of training and testing rows
train, test = partition(eachindex(seeds_features_min.Result), 0.7, shuffle=true)
validate = [size(fdata, 1):size(fullY, 1)...]

# Recode result to win/ loss
y = @pipe categorical(fullY) |> recode(_, 0=>"lose",1=>"win");
tree_model = @load DecisionTreeClassifier verbosity=1
tree = machine(tree_model, fullX, y)

@load SVC pkg=LIBSVM

svc_mdl = SVC()
svc = machine(svc_mdl, fullX, fullY)


# Train the model!
fit!(tree, rows = train)
yhat = predict(tree, rows=test)

# evaluate accuracy, cross-entropy
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(tree, rows=test), y[test])

# make the submission prediction
final_prediction = predict_mode(tree, rows=validate)
