from utils import *
from results_heuristics import * 

users_and_facs_df, travel_dict, users, facs = load_an_instance(2)

# results = get_results_open_greedy(users_and_facs_df, travel_dict, users, facs, 
#                             budget_factors = [0.5, 0.1], 
#                             cap_factor=1.5, threads=1, tolerance=0.0,
#                             cutoff_localsearch = 0.2, num_consider_in_relaxation = 50, 
#                             num_consider_per_iterations = [5,50], time_limit_relaxation=1000,
#                             assignment_methods =["greedy_assign"],local_search_methods = ["local_random_reassign"], 
#                             time_limit_while_localsearch = 1, num_fix_assignment_iterations = [len(facs)],
#                             depths_greedy_reassign = [1,2],
#                             final_assignment_methods = ["relaxation_rounding"],output_filename = "open_greedy_instance_1.json")

write_results_open_greedy_table("open_greedy_instance_1.json", "open_greedy_instance_1.xlsx","instance_1_BFLP_MIP.json")
