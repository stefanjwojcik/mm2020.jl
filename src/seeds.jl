"""
This file is responsible for creating basic ncaa seeds and for creating the outcome 'result'
"""

function seed_to_int(seed::String)
	#Get just the digits from the seeding. Return as int
	parse(Int, seed[2:3])
end

#data_dir = '../input/'
def make_seeds():
	print("loading data..")
	df_seeds = load("data/DataFiles/NCAATourneySeeds.csv") |> DataFrame
	df_tour = load("data/DataFiles/NCAATourneyCompactResults.csv") |> DataFrame

	df_seeds.seed_int = seed_to_int.(df_seeds.Seed)
	deletecols!(df_seeds, :Seed)

	deletecols!(df_tour, [:DayNum, :WScore, :LScore, :WLoc, :NumOT])

	print("creating seeds...")
	df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
	df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
	df_winseeds = copy(df_seeds)
	df_lossseeds = copy(df_seeds)
	rename!(df_winseeds, :WTeamID => :TeamID, :WSeed)

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
