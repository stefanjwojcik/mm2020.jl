# calculate season elo rankings from here: https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings
"""
This file is responsible for creating seasonal elo rankings as a predictive feature
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

#def elo_pred(elo1, elo2):
#	return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

function elo_pred(elo1::Float64, elo2::Float64)
	1.0 / (10.0 ^ (-(elo1 - elo2) / 400.0) + 1.0)
end

#def expected_margin(elo_diff):
#	return((7.5 + 0.006 * elo_diff))

function expected_margin(elo_diff)
	(7.5 + 0.006 * elo_diff)
end

#def elo_update(w_elo, l_elo, margin):
#	K = 20.
#	elo_diff = w_elo - l_elo
#	pred = elo_pred(w_elo, l_elo)
#	mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
#	update = K * mult * (1 - pred)
#	return(pred, update)

function elo_update(w_elo, l_elo, margin)
	K = 20.0
	elo_diff = w_elo - l_elo
	pred = elo_pred(w_elo, l_elo)
	mult = ((margin + 3.0)^0.8) / expected_margin(elo_diff)
	update = K * mult * (1 - pred)
	pred, update
end

# FINAL ELO FOR THE SEASON: NEED TO LOOK IN WINNER OR LOSER POSITION

#def final_elo_per_season(df, team_id):
#	d = df.copy()
#	d = d.loc[(d.WTeamID == team_id) | (d.LTeamID == team_id), :]
#	d.sort_values(['Season', 'DayNum'], inplace=True)
#	d.drop_duplicates(['Season'], keep='last', inplace=True)
#	w_mask = d.WTeamID == team_id
#	l_mask = d.LTeamID == team_id
#	d['season_elo'] = None
#	d.loc[w_mask, 'season_elo'] = d.loc[w_mask, 'w_elo']
#	d.loc[l_mask, 'season_elo'] = d.loc[l_mask, 'l_elo']
#	out = pd.DataFrame({
#		'team_id': team_id,
#		'season': d.Season,
#		'season_elo': d.season_elo
#	})
#	return(out)

function final_elo_per_season(df, team_id)
	d = copy(df)
	d = d[ (d.WTeamID .== team_id | d.LTeamID .== team_id ), : ]
	sort!(d, (:Season, :DayNum), rev=true)
	unique!(d, :Season)
	w_mask = d.WteamID .== team_id
	l_mask = d.LteamID .== team_id
	d.season_elo = 0.0
	d.season_elo[w_mask] .= d.w_elo[w_mask]
	d.season_elo[l_mask] .= d.l_elo[l_mask]
	out = DataFrame(team_id = team_id, season = d.Season, season_elo = d.season_elo)

end

class elo:
	def __init__(self):
		self.data_path = "data/DataFiles/RegularSeasonCompactResults.csv"
		self.rs = pd.read_csv(self.data_path)
		self.HOME_ADVANTAGE = 100.
		#self.K = 20.
		self.team_ids = set(self.rs.WTeamID).union(set(self.rs.LTeamID))
		# This dictionary will be used as a lookup for current
		# scores while the algorithm is iterating through each game
		self.elo_dict = dict(zip(list(self.team_ids), [1500] * len(self.team_ids)))
		self.rs['margin'] = self.rs.WScore - self.rs.LScore

mutable struct elo
	data_path::String
	rs::DataFrame
	HOME_ADVANTAGE::Float64
	team_ids::Array{String, 1}
	elo_dict::Dict
end
	# I'm going to iterate over the games dataframe using
	# index numbers, so want to check that nothing is out
	# of order before I do that.

function iterate_games(elo_obj::elo)

	# update the basic info
	elo_obj.data_path = "data/DataFiles/RegularSeasonCompactResults.csv"
	elo_obj.rs = load(data_path) |> DataFrame
	elo_obj.HOME_ADVANTAGE = 100.0
	elo_obj.team_ids

	preds = []
	w_elo = []
	l_elo = []

	print("looping over every game in the dataframe")
	# Loop over all rows of the games dataframe
	#for row in self.rs.itertuples():

	for row in eachrow(elo_obj.rs)
		w = row.WTeamID
		l = row.LTeamID
		margin = row.margin
		wloc = row.WLoc

		# Does either team get a home-court advantage?
		w_ad, l_ad, = 0., 0.
		if wloc == "H"
			w_ad += elo_obj.HOME_ADVANTAGE
		elseif wloc == "A"
			l_ad += elo_obj.HOME_ADVANTAGE
		end
		# Get elo updates as a result of the game
		pred, update = elo_update(elo_obj.elo_dict[w] + w_ad,
								  elo_obj.elo_dict[l] + l_ad,
								  margin)
		elo_obj.elo_dict[w] += update
		elo_obj.elo_dict[l] -= update

		# Save prediction and new Elos for each round
		push!(preds, pred)
		push!(w_elo, elo_obj.elo_dict[w])
		push!(l_elo, elo_obj.elo_dict[l])
	end

	elo_obj.rs.w_elo = w_elo
	elo_obj.rs.l_elo = l_elo
	print("done")
	return elo_obj
end

	def elo_ranks(self):
		# load data
		rs = self.iterate_games()
		print("computing elo for each team..")
		df_list = [final_elo_per_season(rs, id) for id in self.team_ids]
		season_elos = pd.concat(df_list)
		# create difference scores
		df_winelo = season_elos.rename(columns={'team_id':'WTeamID', 'season':'Season', 'season_elo': 'W_elo'})
		df_losselo = season_elos.rename(columns={'team_id':'LTeamID', 'season':'Season', 'season_elo': 'L_elo'})
		# Merge in the compact results
		df_tour = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')
		df_dummy = pd.merge(left=df_tour, right=df_winelo, how='left', on=['Season', 'WTeamID'])
		df_concat = pd.merge(left=df_dummy, right=df_losselo, on=['Season', 'LTeamID'])
		df_concat['Elo_diff'] = df_concat.W_elo - df_concat.L_elo
		df_concat.drop(['W_elo', 'L_elo'], axis=1, inplace = True)

		df_wins = pd.DataFrame()
		df_wins = df_concat[['Season',  'WTeamID',  'LTeamID', 'Elo_diff']]
		df_wins['Result'] = 1


		df_losses = pd.DataFrame()
		df_losses = df_concat[['Season',  'WTeamID',  'LTeamID']]
		df_losses['Elo_diff'] = -df_concat['Elo_diff']
		df_losses['Result'] = 0

		df_out = pd.concat((df_wins, df_losses))

		print("done")
		return(df_out)

#season_elos.sample(10)

#season_elos.to_csv("season_elos.csv", index=None)
