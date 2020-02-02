"""
This file is responsible for creating basic ncaa seeds and for creating the outcome 'result'
"""
using DataFrames, CSVFiles

#data_dir = '../input/'
function historic_matchups(df_path = "data/DataFiles/NCAATourneyCompactResults.csv"):
	print("loading data..")
	df_tour = load(df_path) |> DataFrame;
	df_tour.Diff_HistScore = df_tour.WScore - df_tour.LScore
	deletecols!(df_tour, [:DayNum, :WScore, :LScore, :WLoc, :NumOT])

	print("creating historical matchups...")
	df_win = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
	df_win = copy(df_tour)
	rename!(df_win, :WTeamID => :TeamID, :WSeed => :seed_int)
	altnames = [Symbol(replace(String(x), "W" => "")) for x in names(df_tour)]
	df_losss = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
	df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
	df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
	df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed

	df_wins = pd.DataFrame()
	df_wins = df_concat[['Season',  'WTeamID',  'LTeamID', 'SeedDiff']]
	df_wins['Result'] = 1

	df_losses = pd.DataFrame()
	df_losses = df_concat[['Season',  'WTeamID',  'LTeamID']]
	df_losses['SeedDiff'] = -df_concat['SeedDiff']
	df_losses['Result'] = 0

	df_predictions = pd.concat((df_wins, df_losses))
	print("done")
	return(df_predictions)
