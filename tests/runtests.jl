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

##############################################################
# Create training features for valid historical data
# seeds
seeds_features = make_seeds()
# efficiency
Wfdat, Lfdat, effdat = eff_stat_seasonal_means()
eff_features = get_eff_tourney_diffs(Wfdat, Lfdat, effdat)
# ELO
season_elos = elo_ranks(Elo())
elo_features = get_elo_tourney_diffs(season_elos)
### Loading the basic seeds data

# Create features required to make submission predictions
seed_submission = get_seed_submission_diffs(submission_sample, seeds_df)
eff_submission = get_eff_submission_diffs(submission_sample, effdat) #see above
elo_submission = get_elo_submission_diffs(submission_sample, season_elos)

# Submission data
subsample = load("/home/swojcik/github/mm2020.jl/data/MSampleSubmissionStage1_2020.csv") |> DataFrame;
seeds_df = load("/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MNCAATourneySeeds.csv") |> DataFrame;
submission_df = gen_seed_features(subsample, seeds_df);

# Join the two feature sets
featurecols = [:SeedDiff]
fullX = [seeds[featurecols]; submission_df[featurecols]]
fullY = [seeds.Result; repeat([0], size(submission_df, 1))]

# create array of training and testing rows
train, test = partition(eachindex(ncaa_df.Result), 0.7, shuffle=true)
validate = [size(ncaa_df, 1):size(fullY, 1)...]

# Recode result to win/ loss
y = @pipe categorical(fullY) |> recode(_, 0=>"lose",1=>"win");
tree_model = @load DecisionTreeClassifier verbosity=1
tree = machine(tree_model, fullX, y)

# Train the model!
fit!(tree, rows = train)
yhat = predict(tree, rows=test)

# evaluate accuracy, cross-entropy
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(tree, rows=test), y[test])

# make the submission prediction
final_prediction = predict_mode(tree, rows=validate)
