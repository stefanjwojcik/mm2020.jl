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

export  make_seeds,
        get_seed_submission_diffs,
        get_eff_tourney_diffs,
        get_eff_submission_diffs,
        Elo,
        get_elo_tourney_diffs,
        get_elo_submission_diffs


##############################################################################
##
## Load files
##
##############################################################################

include("efficiency.jl")
include("seeds.jl")
include("elo.jl")

end # module
