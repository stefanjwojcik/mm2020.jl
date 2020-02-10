"""
This file is responsible for creating basic ncaa seeds and for creating the outcome 'result'
"""

function seed_to_int(seed::String)
	#Get just the digits from the seeding. Return as int
	parse(Int, seed[2:3])
end

#data_dir = '../input/'
function make_seeds()

	print("loading data..")
	df_seeds = load("data/DataFiles/NCAATourneySeeds.csv") |> DataFrame
	df_tour = load("data/DataFiles/NCAATourneyCompactResults.csv") |> DataFrame

	df_seeds.seed_int = seed_to_int.(df_seeds.Seed)

	deletecols!(df_seeds, :Seed)
	deletecols!(df_tour, [:DayNum, :WScore, :LScore, :WLoc, :NumOT])

	print("creating seeds...")

	df_winseeds = copy(df_seeds)
	df_lossseeds = copy(df_seeds)

	rename!(df_winseeds, :TeamID => :WTeamID, :seed_int => :WSeed)
	rename!(df_lossseeds, :TeamID => :LTeamID, :seed_int => :LSeed)

	df_dummy = join(df_tour, df_winseeds, on = [:Season, :WTeamID], kind = :left)
	df_concat = join(df_dummy, df_lossseeds, on = [:Season, :LTeamID])
	df_concat.SeedDiff = df_concat.WSeed - df_concat.LSeed

	df_wins = copy(df_concat[[:Season, :WTeamID, :LTeamID, :SeedDiff]])
	df_wins.Result = 1

	df_losses = copy(df_concat[[:Season, :WTeamID, :LTeamID]])
	df_losses.SeedDiff = df_concat.SeedDiff*-1
	df_losses.Result = 0

	println("done")
	df_predictions = [df_wins; df_losses]

end
