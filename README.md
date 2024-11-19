# Heuristics for a capacitated facility location problem with a quadratic objective function
This project contains the code used within  by M. Schmidt's master thesis and the paper **Heuristics for a capacitated facility location problem with a quadratic objective function** by M. Schmidt and B. Singh. We provide the implementations of all the algorithms discussed in **Heuristics for a capacitated facility location problem with a quadratic objective function** and also to the instances we used to test our heuristics on.

## Repository content
The repository is structure as follows:
- ```data```: This folder contains the data for four instances. For each instance, there is a file containing the travel probabilities $P_{ij}$ (e.g. ```instance_1_travel_dict.json.pbz2```) and a file containing the rest of the input data in a dataframe (e.g. ```instance_1_users_and_facs.csv```). For each instance, we also provide a corresponding instance where the capacity has been scaled to $C_j = \sum_{i \in I} U_i P_{ij}$ (```instance_1_suff_users_and_facs.csv```). To load the data, the function ```load_an_instance``` is provided. An example of loading Instance 1 without sufficient capacity can be seen below. ```users``` is the set $I$ and ```facs``` is the set $J$. Note that in our use of the instances we scale the input capacities by a factor. Whenver a ```cap_factor``` is an input the a function, set this to 1.0 if using any of the sufficient capacity instances, 1.5 if using Instance 1 or Instance 2 and 0.8 if using Instance 3 or Instance 4.
```python
users_and_facs_df, travel_dict, users, facs = load_an_instance(1, False)
```
- ```heuristics_and_mips```: This folder contains the main part of the code. We will briefly describe the purpose of each of the files:
	- ```BFLP_heuristics.py```: This file contains the code for all the BFLP heurisitcs. 
	- ```BFLP_MIP.py```: This file contains the code for running the BFLP MIP.
	- ```BUAP_heuristics.py```: This file contains all the BUAP heuristics and also implementations of those that are just needed for use within the BFLP heuristics, e.g. the optimisation of relaxation rounding when used in the first version of close greedy.
	- ```BUAP_MIP.py```: This file contains the code for running the BUAP MIP and also for editing the model, as needed for relaxation rouding with the first version of close greedy.
    - ```main.py```: This file contains an example of how to use the functions in the repository.
	- ```results_heuristics.py```: This file contains functions for running all the heuristics across multiple budget factors and input parameters. For each BFLP heuristic, a function for running the heuristics with multiple inputs (function name starting with get_) exists, resulting in a json, and a function for converting these results into more easily readable excel tables (function name starting with write_). Any results created here will be saved in the own_results folder, which will be created if it does not exist yet. For the BUAP, there is one function that runs all the heuristics on a single instance and the corresponding function which writes the results into a table.
	- ```utils.py```: This file contains a few subroutines that are useful for reading and writing data, and some subroutines that are used in multiple heuristics. 

## Example

We provide a short code example of how to use this code in ```main.py```. To run the file, make sure you have navigated to the repository folder in your terminal. Then, you can run the file using ```python3 ./heuristics_and_mips/main.py```.
In this example, we run the open greedy algorithm on Instance 1 to test out how different values for the parameters $n_c$ and $d$ perform across different budgets. This is assuming that we have the results for the BFLP MIP for this instance saved in the file ```instance_1_BFLP_MIP.json``` in the own_results folder. 


## Requirements for running the code
Before cloning the code, make sure you have [Git Large File Storage (Git LFS)](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) installed. Please see instructions for downloading and installing Git LFS on the linked website. Note that the data files will not function properly, due to their size, if you just download the zipped file. After installing Git LFS, clone the repository as usual (with `git clone`) to get the actual data files. Then, navigate into the required folder as usual (with `cd`) and pull the large data files with Git LFS (`git lfs pull`)
 
Further, before running the code, make sure you have the below installed and linked with their respective paths (see instructions on installation on their respective pages):
- python3
- numpy
- pandas
- geopy
- pyomo
- gurobi
