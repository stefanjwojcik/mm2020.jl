__precompile__()

module mm2020

##############################################################################
##
## Dependencies
##
##############################################################################

using CSVFiles, DataFrames, Statistics

##############################################################################
##
## Exported methods and types
##
##############################################################################

export  get_eff_tourney_diffs,
        get_eff_submission_diffs,
        make_seeds,
        gen_seed_features,
        Elo,
        get_elo_tourney_diffs,
        get_elo_sub_diffs


##############################################################################
##
## Load files
##
##############################################################################

include("efficiency.jl")
include("seeds.jl")
include("elo.jl")

end # module
