# Heuristics for a capacitated facility location problem with a quadratic objective function
This project contains the code used within  by M. Schmidt's master thesis and the paper **Heuristics for a capacitated facility location problem with a quadratic objective function** by M. Schmidt and B. Singh. We provide the implementations of all the algorithms discussed in **Heuristics for a capacitated facility location problem with a quadratic objective function** and also to the instances we used to test our heuristics on.

## Repository content
The repository is structure as follows:
- data: This folder contains the data for four instances. For each instance, there is a file containing the travel probabilities P_ij (e.g. ‘instance_1_travel_dict.json.pbz2’) and a file containing the rest of the input data in a dataframe (e.g. ‘instance_1_users_and_facs.csv’). For each instance, we also provide a corresponding instance where the capacity has been scaled to C_j = sum_{i \in I} U_i P_ij (‘instance_1_suff_users_and_facs.csv’). To load the data, the function ‘load_an_instance’ is provided. An example of loading instance 1 without sufficient capacity can be seen below. ‘users’ is the set I and ‘facs’ is the set J. Note that in our use of the instances we scale the input capacities by a factor. Whenver a ‘cap_factor’ is an input the a function, set this to 1.0 if using any of the sufficient capacity instances, 1.5 if using Instance 1 or Instance 2 and 0.8 if using Instance 3 or Instance 4.
```python
users_and_facs_df, travel_dict, users, facs = load_an_instance(1, False)
```
- heuristics_and_mips: This folder contains the main part of the code. We will briefly describe the purpose of each of the files:
	- BFLP_heuristics.py: This file contains the code for all the BFLP heurisitcs. 
	- BFLP_MIP.py: This file contains the code for running the BFLP MIP.
	- BUAP_heuristics.py: This file contains all the BUAP heuristics and also implementations of those that are just needed for use within the BFLP heuristics, e.g. the optimisation of relaxation rounding when used in the first version of close greedy.
	- BUAP_MIP.py: This file contains the code for running the BUAP MIP and also for editing the model, as needed for relaxation rouding with the first version of close greedy.
    - main.py: This file contains an example of how to use the functions in the repository.
	- results_heuristics.py: This file contains functions for running all the heuristics across multiple budget factors and input parameters. For each BFLP heuristic, a function for running the heuristics with multiple inputs (function name starting with get_) exists, resulting in a json, and a function for converting these results into more easily readable excel tables (function name starting with write_). Any results created here will be saved in the own_results folder, which will be created if it does not exist yet. For the BUAP, there is one function that runs all the heuristics on a single instance and the corresponding function which writes the results into a table.
	- utils.py: This file contains a few subroutines that are useful for reading and writing data, and some subroutines that are used in multiple heuristics. 

## Example

We provide a short code example of how to use this code. In this example, we run the open greedy algorithm on Instance 1 to test out how different values for the parameters n_c and d perform across different budgets. This is assuming that we have the results for the BFLP MIP for this instance saved in the file instance_1_BFLP_MIP.json in the own_results folder.

```python
from utils import *
from results_heuristics import * 

users_and_facs_df, travel_dict, users, facs = load_an_instance(1)

results = get_results_open_greedy(users_and_facs_df, travel_dict, users, facs, 
                            budget_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 
                            cap_factor=1.5, threads=1, tolerance=0.0,
                            cutoff_localsearch = 0.2, num_consider_in_relaxation = 50, 
                            num_consider_per_iterations = [5,50], time_limit_relaxation=1000,
                            assignment_methods =["greedy_assign"],local_search_methods = ["local_random_reassign"], 
                            time_limit_while_localsearch = 1, num_fix_assignment_iterations = [len(facs)],
                            depths_greedy_reassign = [1,2],
                            final_assignment_methods = ["relaxation_rounding"],
                            output_filename = "open_greedy_instance_1.json")

write_results_open_greedy_table("open_greedy_instance_1.json", "open_greedy_instance_1.xlsx","instance_1_BFLP_MIP.json")

```

## Requirements for running the code
Apart from some common python packages, the code requires Pyomo and Geopy. In addition, a Gurobi license and installation is required.
