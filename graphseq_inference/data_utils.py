import os, sys, shutil, pickle
from copy import copy
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool, cpu_count



import numpy as np
import pandas as pd
#import seaborn as sns

import msprime
import networkx as nx

import torch
from torch_sparse import SparseTensor
import torch_geometric
from torch_geometric.utils.convert import from_networkx

import tskit
from typing import Union



import tsinfer
import tsdate

def sample_population_size(n_min:int=10, n_max:int=100_000, num_time_windows=21) -> list[float]:
    
    """Creates random demography. Function taken from: 
    https://gitlab.inria.fr/ml_genetics/public/dlpopsize_paper
    
    :param int n_min: Lower-bound of demography.
    :param int n_max: Upper-bound of demography.
    :param int num_time_windows: Number of population sizes in demography.
    :return list: 
    """
    
    n_min_log10 = np.log10(n_min)
    n_max_log10 = np.log10(n_max)
    population_size = [10 ** np.random.uniform(low=n_min_log10, high=n_max_log10)] 
    for j in range(num_time_windows - 1):
        population_size.append(10 ** n_min_log10 - 1)
        while population_size[-1] > 10 ** n_max_log10 or population_size[-1]  < 10 ** n_min_log10:
            population_size[-1] = population_size[-2] * 10 ** np.random.uniform(-1, 1)
            
    return population_size



def sample_population_size_with_varying_lower_bound(lower_bound: int, n_min:int=10, n_max:int=100_000, num_time_windows=21) -> list[float]:
    
    """Creates random demography. Function modified after: 
    https://gitlab.inria.fr/ml_genetics/public/dlpopsize_paper
    Decreasing number of bottlenecks in the dataset by increaseing the lower-bound.
    
    :param int lower_bound: Modified lower-bound of demography.
    :param int n_min: Lower-bound of demography.
    :param int n_max: Upper-bound of demography.
    :param int num_time_windows: Number of population sizes in demography.
    :return list: 
    """
    
    n_min_log10 = np.log10(n_min)
    n_max_log10 = np.log10(n_max)
    population_size = [10 ** np.random.uniform(low=np.log10(lower_bound), high=n_max_log10)] 
    for j in range(num_time_windows - 1):
        population_size.append(lower_bound - 1)
        while population_size[-1] > 10 ** n_max_log10 or population_size[-1]  < lower_bound:
            population_size[-1] = population_size[-2] * 10 ** np.random.uniform(-1, 1)
            
    return population_size


def get_population_time(time_rate:float=0.06, tmax:int = 130_000,
                        num_time_windows:int = 21
                       ) -> np.array :
    """Creates population time points; used as time points to change
    population size changes for simulation
    
    :return numpy.ndarray: time points of length num_time_windows
    """
    
    population_time = np.repeat([(np.exp(np.log(1 + time_rate * tmax) * i /
                              (num_time_windows - 1)) - 1) / time_rate for i in
                              range(num_time_windows)], 1, axis=0)
    population_time[0] = 1
    return population_time


def simulate_scenario(population_size: Union[list, np.ndarray],
                      population_time: Union[list, np.ndarray],
                      mutation_rate: float,
                      recombination_rate: float,
                      segment_length: float,
                      num_sample:int,
                      num_replicates: int,
                      seed: int = 69420,
                      model = None,
                     ):

    """ Simulates tree sequence with msprime given population size changes at specific time-points.
    Piece-wise constant simualtion of demography.
    
    :return: generator of tskit.trees.TreeSequence
    """

    demography=msprime.Demography()
    demography.add_population(initial_size=(population_size[0]))
    for i, (time, size) in enumerate(zip(population_time, population_size)):
        if i != 0:
            demography.add_population_parameters_change(time=time, initial_size=size)

    tss = msprime.sim_ancestry(samples=num_sample, recombination_rate=recombination_rate,
                                          sequence_length=int(segment_length), demography=demography,
                                          ploidy=1, model=model, num_replicates=num_replicates, random_seed=seed)

    return tss



def simulate_scenario_haploid(population_size: Union[list, np.ndarray],
                      population_time: Union[list, np.ndarray],
                      mutation_rate: float,
                      recombination_rate: float,
                      segment_length: float,
                      num_sample:int,
                      num_replicates: int,
                      seed: int = 69420,
                      model = None,
                     ):
    
    """ Simulates tree sequence with msprime given population size changes at specific time-points.
    Piece-wise constant simualtion of demography.
    
    :return: generator of tskit.trees.TreeSequence
    """
    
    demography=msprime.Demography()
    demography.add_population(initial_size=(population_size[0]))
    for i, (time, size) in enumerate(zip(population_time, population_size)):
        if i != 0:
            demography.add_population_parameters_change(time=time, initial_size=size)

    tss = msprime.sim_ancestry(samples=num_sample, recombination_rate=recombination_rate,
                                          sequence_length=int(segment_length), demography=demography,
                                          ploidy=1, model=model, num_replicates=num_replicates, random_seed=seed)

    return tss

def simulate_scenario_diploid(population_size: Union[list, np.ndarray],
                      population_time: Union[list, np.ndarray],
                      mutation_rate: float,
                      recombination_rate: float,
                      segment_length: float,
                      num_sample:int,
                      num_replicates: int,
                      seed: int = 69420,
                      model = None,
                     ):
    
    """ Simulates tree sequence with msprime given population size changes at specific time-points.
    Piece-wise constant simualtion of demography.
    
    :return: generator of tskit.trees.TreeSequence
    """
    
    demographic_events = [msprime.PopulationParametersChange(
                time=population_time[i],
                growth_rate=0,
                initial_size=population_size[i]) for i in range(1, len(population_time))]

    population_configurations = [msprime.PopulationConfiguration(
                sample_size=num_sample,
                initial_size=population_size[0])]

    tss = msprime.simulate(
                length=segment_length,
                population_configurations=population_configurations,
                demographic_events=demographic_events,
                recombination_rate=recombination_rate,
                mutation_rate=mutation_rate,
                num_replicates=num_replicates,
                random_seed=seed, model=model
                )
    return tss


def discretize_trees(trees: list[tskit.trees.Tree], num_trees:int=2000) -> list[tskit.trees.Tree]:
    
    """ Choosing num_trees of sequence length (determined by last trees right interval border);
    thus discretizing abitrary number of trees to fixed size.

    Arg types:
        * **trees** *(list[tskit.trees.Tree])* - Python list of tskit trees.
        * **num_trees** *(int)* - Number of trees after discretization.

    Return types:
        * **trees** *(list[tskit.trees.Tree])* - Python list of tskit trees.

    """
    
    segment_length = trees[-1].interval.right
    intervals = np.linspace(0, segment_length, num_trees)
    discretized_trees = []
    for interval in intervals:
        for tree in trees:
            if interval >= tree.interval.left and interval <= tree.interval.right:
                discretized_trees.append(tree)
    return discretized_trees



def rename_data_attribute(obj, old_name, new_name):
    obj.__dict__['_store'][new_name] = obj.__dict__['_store'].pop(old_name)

    
def datatize_trees(trees: list[tskit.trees.Tree]) -> list[torch_geometric.data.data.Data]:
    """ Converting tskit list of trees to list of torch_geometric data objects. Function also renames
    branch_length, which contains all the edge weight parameters to edge_weight.
    
    Arg types:
        * **trees** *(list[tskit.trees.Tree])* - list of tskit trees.
    Return types:
        * **data_objects** *(list[torch_geometric.data.data.Data])* - list of torch_geometric data objects.
    
    """
    data_objects = [from_networkx(nx.Graph(tree.as_dict_of_dicts())) for tree in trees]
    for data in data_objects: rename_data_attribute(data, "branch_length", "edge_weight") 
    return data_objects



def sparse_datatize_trees(trees: list[tskit.trees.Tree]) -> list[torch_geometric.data.data.Data]:
    """ Converting tskit list of trees to list of sparse torch_geometric data objects. Data objects
    only contain an adjaceny matrix, instead of edge_index and edge_weight tensors.
    
    Arg types:
        * **trees** *(list[tskit.trees.Tree])* - list of tskit trees.
    Return types:
        * **data_objects** *(list[torch_geometric.data.data.Data])* - list of torch_geometric data objects.
    
    """
    
    data_objects = []
    for tree in trees:
        data = from_networkx(nx.Graph(tree.as_dict_of_dicts()))
        edge_index, num_nodes, edge_attr = data.edge_index, data.num_nodes, data.branch_length

        edge_attr = torch.log(edge_attr)

        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr,
                           sparse_sizes=(num_nodes, num_nodes))

        data.adj = adj
        data.edge_index = None
        data.num_nodes = None
        data.branch_length = None
        data_objects.append(data)
        
    return data_objects


def filter_node_times(node_times:list[float] , tree:tskit.trees.Tree)->list[float]:
    """ Filters list of all tree sequence node times to only contain node times of given tree.
    Arg types:
        * **node_times** *(list)* - list containing node times, index 0 contains node time of node 0, index 1 of node 1 and so on.
        * **tree** (tskit tree object) - tskit tree of which node times should be extracted
    Return types:
        * **tree_node_times** - list of node times of given tree
    """
    current_tree_nodes = [node for node in tree.nodes()]
    current_tree_node_times = node_times[current_tree_nodes].tolist()
    return current_tree_node_times


def get_sorted_log_trees_node_times(ts: tskit.trees.TreeSequence, trees: list[tskit.trees.Tree]) -> list[list[float]]:
    
    """ Creates list of list of all node_times for each tree. Times are sorted, natural log-scaled and padded on the right 
    side with last node time value, so all node times are of equal length. Padding important to calculate masks for infered
    trees or non-wright-fisher models. All leave node times are removed, because these are just zero.
    
    Arg types:
        * **ts** *(tskit tree sequence)* - Input simulated or infered tree sequence.
        * **trees** *(list of tskit trees)* - List of trees, e.g. ts.aslist(), discretize_trees(ts.aslist(), num_trees)
    Return types:
        * **output** *(list of list)* - list of list of node times for each tree, sorted, natural log-scaled and right-padded.
    """
    
    output = []
    
    node_times = np.array([node.time for node in ts.nodes()])
    
    num_leave_nodes = ts.get_sample_size()
    num_non_leave_nodes = 2 * num_leave_nodes - 1 - num_leave_nodes
    
    for tree in trees:
        current_tree_node_times = filter_node_times(node_times, tree)
        
        current_tree_node_times = sorted(current_tree_node_times)
        current_tree_node_times = [time for time in current_tree_node_times if time != 0]
        log_current_tree_node_times = np.log(current_tree_node_times).tolist()
                
        while len(log_current_tree_node_times) != num_non_leave_nodes:
            log_current_tree_node_times.append(log_current_tree_node_times[-1])
            
        output.append(log_current_tree_node_times)
        
    return output


def get_population_time_mask(sorted_log_trees_node_times:list[list[float]], log_population_time:np.ndarray, num_time_windows:int=21, hard_lower_threshold=None) -> np.ndarray:
    """ Creates a boolean mask indicating the nounderies where no coalescent event occured for all trees in a tree sequence.
    
    Arg types:
        * **sorted_log_trees_node_times** *(lsit of list)* - list of list of node times for each tree, sorted, natural log-scaled and right-padded.
        * **log_population_time** *(numpy array)* - natural log-scaled population time
    Return types:
        * **mask** *(numpy array)* - boolean mask
    """

    sorted_log_trees_node_times =  np.array(sorted_log_trees_node_times)
    n_trees = sorted_log_trees_node_times.shape[0]

    masked_population_time = np.tile(log_population_time, n_trees)

    last_times = sorted_log_trees_node_times[:,-1]
    first_times = sorted_log_trees_node_times[:,0]
    if hard_lower_threshold is not None:
        first_times[:] = np.log(hard_lower_threshold)
    
    mask = masked_population_time < np.tile(last_times, num_time_windows)
    mask1 = masked_population_time > np.tile(first_times, num_time_windows)
    mask = mask.reshape(n_trees, len(log_population_time))
    mask1 = mask1.reshape(n_trees, len(log_population_time))
    
    return np.logical_and(mask, mask1)




def sample_parameters(num_scenario:int,
                      increment = 0,
                      num_replicates = 50,
                      num_time_windows = 21,
                      n_min = 100,
                      n_max = 100_000,
                      recombination_rates = [1e-9, 1e-8],
                      mutation_rates: list[float] = [1e-9, 1e-8],
                      population_size: list[float] = None,
                      use_varying_lower_bounds: bool = False,
                      lower_bounds: list = None,
                      model = None,
                      alpha = None
                      
                      ) -> pd.DataFrame:
    
    """ Creates (random) parameters for simulation if not list of length num_time_windows is given as population_size parameter.
    
    Arg types:
        * **num_scenario** *(int)* - Number of scenarios to create parameters.
        * **increment** *(int)* - Starting number of scenarios for easier merging incase of using the function multiple times
        * **recombination_rates** *(int)* - List of length two given minium and maxium rate for uniform sampling.
        * **mutation_rates** *(int)* - List of length two given minium and maxium rate for uniform sampling.
        * **population_size** *(int)* - Population sizes.
        * **model** *(float)* - .
    Return types:
        * **parameters** *(pandas dataframe)* - Parameters for simulation
    
    """
    parameter_names = ["scenario", "recombination_rate", "mutation_rate", "rho_theta"]
    for i in range(num_time_windows):
        parameter_names.append("pop_size_" + str(i))
    parameter_names.append("replicate")
    parameter_names.append("model")

    #population_time = get_population_time()

    parameters = []
    
    use_random_population_size_sampling = True
    if population_size is not None:
        use_random_population_size_sampling = False
   
    for i in tqdm(range(increment, num_scenario+increment)):
        
        recombination_rate = np.random.uniform(low=recombination_rates[0], high=recombination_rates[1])
        mutation_rate = np.random.uniform(low=mutation_rates[0], high=mutation_rates[1])
        rho_theta = np.round(recombination_rate/mutation_rate, 2)
        
       
        if model != None: 
            if alpha == None: alpha = np.random.uniform(1.01, 1.99)
            
        
        if use_random_population_size_sampling and not use_varying_lower_bounds:
            population_size = sample_population_size(n_min=n_min, n_max=n_max, num_time_windows=num_time_windows)#[:num_time_window]
        elif use_random_population_size_sampling and use_varying_lower_bounds:
            assert len(lower_bounds) > 0
            lower_bound = np.random.choice(lower_bounds)
            population_size = sample_population_size_with_varying_lower_bound(lower_bound, n_min=n_min, n_max=n_max, num_time_windows=num_time_windows)#[:num_time_window]
            
        
        parameter = [i, recombination_rate, mutation_rate, rho_theta]
        for current_population_size in population_size: parameter.append(current_population_size)
        
            
        for replicate in range(num_replicates):
            parameter_replicate = copy(parameter)
            parameter_replicate.append(str(replicate))
            parameter_replicate.append(alpha)
            parameters.append( parameter_replicate )
            
        
            
    parameters = pd.DataFrame(parameters, columns=parameter_names)
    assert parameters.shape[0] == num_replicates*num_scenario
    return parameters



def simulate_tree_sequence(parameters: pd.DataFrame,
                           population_time: list, 
                           segment_length = 1e6, 
                           num_sample = 10, 
                           num_replicates = 100,
                           seed = 69420,
                          ) -> list[tskit.trees.TreeSequence]:
    
    """ Simulate tree sequences from parameter file and returns them as list.
    
    Arg types:
        * **parameters** *(int)* - Parameters from sample_parameters() function.
        * **population_time** *(int)* - List of population times.
        * **segment_length** *(int)* - Segment_length
        * **num_sample** *(int)* - Number of samples.
        * **num_replicates** *(int)* - Number of replicates.
        * **seed** *(int)* - Seed.
    Return types:
        * **tree_sequences** *(list of tree sequences)* 
    
    """
    
    tree_sequences = []
    for i in tqdm(range(0, parameters.shape[0], num_replicates)):
        population_size = parameters.iloc[i]["pop_size_0":"pop_size_" + str(len(population_time)-1)].tolist()
        mutation_rate = parameters.iloc[i]["mutation_rate"]
        recombination_rate = parameters.iloc[i]["recombination_rate"]
        alpha = parameters.iloc[i]["model"]
        
        model = None
        if alpha != "wf": model = msprime.BetaCoalescent(alpha=alpha)
        
        
        tss = simulate_scenario(population_size=population_size,
                        population_time=population_time,
                        mutation_rate=0, # otherwise memory not sufficient
                        recombination_rate=recombination_rate,
                        segment_length=segment_length,
                        num_sample=num_sample,
                        num_replicates=num_replicates,
                        seed=seed, model=model)
        
        for ts in tss: tree_sequences.append(ts)
        
    return tree_sequences
        
    
    
def extract_parameter(parameter: np.ndarray, pop_size_0_idx=4, pop_size_last_idx=20)->tuple[float, float, list[float], int, int]:
    
    """ Extract single parameter set from parameter row of parameters dataframe.
    Arg types:
        **paramter** *(numpy array)* - Paramter row of parameters dataframe.
    """
    
    scenario_idx = 0
    recombination_rate_idx = 1
    mutation_rate_idx = 2
    pop_size_0_idx = 4
    replicate_idx = pop_size_last_idx+1
    
    recombination_rate = parameter[recombination_rate_idx]
    mutation_rate = parameter[mutation_rate_idx]
    population_size = parameter[pop_size_0_idx:pop_size_last_idx+1]
    scenario = parameter[scenario_idx]
    replicate = parameter[replicate_idx]
    
    return recombination_rate, mutation_rate, population_size, scenario, replicate

def convert_tree_sequence_to_data_object(tree_sequence: tskit.trees.TreeSequence,
                                         parameter: np.ndarray,
                                         population_time: list,
                                         num_trees:int = 500,
                                         num_embedding:int = 19, 
                                         directory: str = "datasets",
                                         hard_lower_threshold = None
                           ):
    """ Converts tree_sequence to a data_object.
    
    
    Arg types:
        * **tree_sequence** *(tree sequence)* - A single tree sequence
        * **parameter** *(numpy array)* - Parameter for simulation.
        * **population_time** *(list of floats)* - Population times.
        * **num_trees** *(int)* - Number of trees used for discretization.
        * **num_embedding** *(int)* - emdbedding.
    
    """
    
    recombination_rate, mutation_rate, population_size, scenario, replicate = extract_parameter(parameter)    
    y = torch.Tensor(population_size)
    trees =  discretize_trees(tree_sequence.aslist(), num_trees = num_trees)
    mask = get_population_time_mask(get_sorted_log_trees_node_times(tree_sequence, trees), np.log(population_time), hard_lower_threshold=hard_lower_threshold)
    data_objects = datatize_trees(trees)

    max_num_nodes = 2 * tree_sequence.num_samples - 1 
    for _ , data in enumerate(data_objects):
        num_nodes = data.num_nodes
        data.x = torch.eye(max_num_nodes,num_embedding)
        data.x[num_nodes:] = torch.zeros(num_embedding)
        data.y = torch.Tensor(np.log(population_size))
        data.num_nodes = max_num_nodes

    torch.save((data_objects, mask), open("./" + str(directory) + "/data_" + str(scenario) + "_" + str(replicate) + ".pth", "wb"))

def convert_tree_sequences_to_data_objects(tree_sequences: list[tskit.trees.TreeSequence],
                                           parameters: np.ndarray,
                                           population_time: np.ndarray,
                                           num_trees:int = 500,
                                           num_embedding:int = 19, 
                                           directory: str = "datasets",
                                           num_cpus: int = 1,
                                           hard_lower_threshold = None,
                           ):
    """ Converts tree_sequences to a data_objects.
    
    
    Arg types:
        * **tree_sequences** *(list of tree sequences)* - A list of tree sequences.
        * **parameter** *(numpy array)* - Parameter for simulation.
        * **population_time** *(numpy array)* - Population times.
        * **num_trees** *(int)* - Number of trees used for discretization.
        * **num_embedding** *(int)* - emdbedding.
        * **directory** *(str)* - Output directory.
    
    """
    if not os.path.exists(directory): os.makedirs(directory)

    
    args = []
    for ts, parameter in zip(tree_sequences, np.array(parameters)):
        args.append((ts, parameter.tolist(), population_time, num_trees, num_embedding, directory, hard_lower_threshold))

    with Pool(num_cpus) as p: 
        _ = p.starmap(convert_tree_sequence_to_data_object, tqdm(args, total=len(tree_sequences)))




def num_mutation_of_tree_sequence(tree_sequence): return tree_sequence.num_mutations

def filter_min_mutations(min_mutations, tree_sequences, parameters, num_replicates, num_cpus):
    
    """ Filtering tree sequences by number of mutations.
    Arg types:
        * **min_mutations** *(int)* - Minimum number of mutations.a replicate must have.
        * **tree_sequences** *(list of tree sequences)* - A list of tree sequences.
        * **parameter** *(numpy array)* - Parameter for simulation.
        * **num_replicates** *(int)* - Number of replicates.
        * **num_cpus** *(int)* - Number of cpus to use.
    """

    with Pool(num_cpus) as p:
        num_mutations = list(tqdm(p.imap(num_mutation_of_tree_sequence, tree_sequences), total=len(tree_sequences)))

        
    
    mask = np.array(num_mutations) > min_mutations

    for i in range(0, parameters.shape[0], num_replicates):
        replicate_indices = list(range(i, i+num_replicates))
        if mask[replicate_indices].sum(0) != num_replicates:
            mask[replicate_indices] = False
    tree_sequences = (np.array(tree_sequences)[mask]).tolist()
    parameters = parameters[mask]
    parameters.reset_index(drop=True, inplace=True)
    
    return tree_sequences, parameters, num_mutations


def uniformize_mask(mask: np.ndarray, threshold:int = 50):
    
    """ Uniformize a copy of a population time mask. Entire column for that tree
    will be either true or false, based on threshold value.
    
    Arg types:
        * **mask** *(np.ndarray)* - Mask
        * **threshold** *(int)* - minimum number of trees having coalescent event during that time.
    
    Return types:
        * **mask** *np.ndarray* - Copy of mask.
    
    """
    
    mask = deepcopy(mask)
    mask[:, mask.sum(0) < threshold] = False
    mask[:, mask.sum(0) >= threshold] = True
    return mask



def get_length_positive_mask_part(tree_sequence, population_time, num_trees):
    
    """ Get length of mask after discretization and uniformization. 
    """
    trees =  discretize_trees(tree_sequence.aslist(), num_trees = num_trees)
    mask = get_population_time_mask(get_sorted_log_trees_node_times(tree_sequence, trees), np.log(population_time))
    mask = uniformize_mask(mask)
    return mask[0][mask[0]].shape[0]

def filter_min_time_length(min_length, tree_sequences, parameters, population_time, num_trees, num_replicates, num_cpus):
    
    """ Filter mask must span at least a coalescent range of min_length time points.
    Arg types:
        * **min_length** *(int)* - Minimum length of consecutive time points.
        * **tree_sequences** *(list of tree sequences)* - A list of tree sequences.
        * **parameter** *(numpy array)* - Parameter for simulation.
        * **num_replicates** *(int)* - Number of replicates.
        * **num_cpus** *(int)* - Number of cpus to use.
    """

    args = []
    for tree_sequence in tree_sequences:
        args.append((tree_sequence, population_time, num_trees))
    
    with Pool(num_cpus) as p:
        postive_mask_lengths = p.starmap(get_length_positive_mask_part, tqdm(args, total=len(args)))

    mask = np.array(postive_mask_lengths) > min_length

    for i in range(0, parameters.shape[0], num_replicates):
        replicate_indices = list(range(i, i+num_replicates))
        if mask[replicate_indices].sum(0) != num_replicates:
            mask[replicate_indices] = False
    tree_sequences = (np.array(tree_sequences)[mask]).tolist()
    parameters = parameters[mask]
    parameters.reset_index(drop=True, inplace=True)
    
    return tree_sequences, parameters, postive_mask_lengths







def infer_tree_sequence(tree_sequences: list[tskit.trees.TreeSequence],
                        parameters: pd.DataFrame,
                        tree_sequence_parameter_index: int,
                        population_time: list[float]) -> tskit.trees.TreeSequence:
    """ Infers tree sequence from tree sequence. Function is written in a way so it becomes easy to use multiprocessing.
    Mutation rate and initial population size are extracted from parameters data frame.

    Arg types:
        * **tree_sequences** *(list of tskit tree sequence)* - List of tree sequences.
        * **parameters** *(int)* - Parameters for simulation.
        * **tree_sequence_parameter_index** *(int)* - Index to chose tree sequence and parameters, respectively.
        * **population_time** *(int)* - Population times.
    Return types:
        * **ts** *(tskit tree sequenc)* - Infered tree sequence.
    """
    
    i = tree_sequence_parameter_index
    ts = tree_sequences[i]

    sample_data = tsinfer.SampleData.from_tree_sequence(ts, 
                                                        use_sites_time=False, 
                                                        use_individuals_time=False)

    inferred_ts = (tsinfer.infer(sample_data)).simplify()
    pop_size0 = parameters.iloc[i]["pop_size_0"]
    mutation_rate = parameters.iloc[i]["mutation_rate"]
    ts = tsdate.date(inferred_ts,
                     Ne = pop_size0,
                     mutation_rate = mutation_rate)
    
    return ts



def infer_tree_sequence(tree_sequence: tskit.trees.TreeSequence,
                        parameter: np.ndarray,
                        ):
    """ Infers tree sequence from tree sequence and parameter row.
    
    Arg types:
        * **tree_sequence** *(tree sequence)* - A single tree sequence
        * **parameter** *(numpy array)* - Parameter for simulation.
    Return types:
        * **tree_sequence** *(tree sequence)* - A single tree sequence
    """
    
    recombination_rate, mutation_rate, population_size, scenario, replicate = extract_parameter(parameter)    
    sample_data = tsinfer.SampleData.from_tree_sequence(tree_sequence, 
                                                        use_sites_time=False, 
                                                        use_individuals_time=False)

    inferred_ts = (tsinfer.infer(sample_data)).simplify()
    pop_size0 = population_size[0]
    tree_sequence = tsdate.date(inferred_ts, Ne = pop_size0, mutation_rate = mutation_rate)
    return tree_sequence

  

def infer_tree_sequences(tree_sequences: tskit.trees.TreeSequence,
                         parameters: pd.DataFrame,
                         num_cpus: int,
                        ):
    """ Infers tree sequences from tree sequences and parameters row.
    
    Arg types:
        * **tree_sequence** *(list tree sequence)* - A list tree sequences
        * **parameter** *(pandas Dataframe)* - Parameter for simulation.
    Return types:
        * **infered_tree_sequences** *(tree sequence)* - A list tree sequences
    """
    
    args = []
    for ts, parameter in zip(tree_sequences, np.array(parameters)):
        args.append((ts, parameter.tolist()))

    with Pool(num_cpus) as p: 
        infered_tree_sequences = p.starmap(infer_tree_sequence, tqdm(args, total=len(tree_sequences)))
        
    return infered_tree_sequences
   






    
    
    
    
    
from collections import Counter
def expand_coalescent_lookup(coalescent_time_lookup: dict, num_time_windows):
    for i in range(num_time_windows):
        if i not in coalescent_time_lookup.keys():
            coalescent_time_lookup[i] = 0

            
def discretize_trees_mmc_version(ts, min_num_nodes = 17, num_trees = 500):
    """ Filters out every tree that has less than min_num_nodes in a tree and 
    samples num_trees with replacement from the remaining trees. 
    """
    trees = ts.aslist()
    num_nodes = [tree_num_nodes(tree) for tree in ts.trees()]
    mask = np.array(num_nodes) >= min_num_nodes
    trees = np.array(trees)[mask]
    trees = np.random.choice(trees, size=num_trees)
    return trees.tolist()
            
def get_binned_coalescent_times(ts, binned_population_time, num_time_windows:int,  num_trees: int = 500) -> list:
    """ Number of coalescent events for each time window for discretized trees.
    """
        
    binned_population_time = np.array(binned_population_time)
    #sorted_log_trees_node_times = get_sorted_log_trees_node_times(ts, discretize_trees(ts.aslist(), num_trees))
    sorted_log_trees_node_times = get_sorted_log_trees_node_times(ts, ts.aslist()[:num_trees])
    
    #sorted_log_trees_node_times = get_sorted_log_trees_node_times(ts, discretize_trees_mmc_version(ts, 17, num_trees))

    tree_times = np.exp(np.array(sorted_log_trees_node_times))
        
    tree_bins = [] 
    outside_window = 0
    
    for current_tree_times in tree_times:
                
        for i, time in enumerate(current_tree_times):
            if time >= 10_000_000:
                outside_window += 1
                time_window = binned_population_time.shape[0]
                tree_bins.append(time_window)
            elif time < 1:
                outside_window += 1
                tree_bins.append(0)
            else:
                #print(time)
                #print(binned_population_time)
                time_window = np.argwhere(np.sum(binned_population_time < time, axis=1) == 1).item()
                tree_bins.append(time_window)
            
    coalescent_times = sorted(np.array(tree_bins).flatten().tolist())
    coalescent_time_lookup = dict(Counter(coalescent_times))
    expand_coalescent_lookup(coalescent_time_lookup, num_time_windows=num_time_windows)
    coalescent_times = [item[1] for item in sorted(coalescent_time_lookup.items())]
    
    return coalescent_times



def uniformize_mask_with_hacky_heuristic(mask, num_time_windows=50, num_replicates=100):

    
    # first heuristic: choosing mask based sliding window
    column_wise_mask = mask.sum(0)
    copied_mask = deepcopy(mask)
    copied_mask[:] = False

    
    pos0 = column_wise_mask[0] == num_replicates
    pos1 = column_wise_mask[1] == num_replicates
    pos2 = column_wise_mask[2] == num_replicates
    pos3 = column_wise_mask[3] == num_replicates
    pos4 = column_wise_mask[4] == num_replicates
    pos5 = column_wise_mask[5] == num_replicates
    
    if pos2 and pos3 and pos4:
        copied_mask[:,1] = True
    if pos3 and pos4 and pos5:
        copied_mask[:,2] = True
    
    
    for i in range(3, num_time_windows-3):

        left_one = column_wise_mask[i-1] == num_replicates
        left_two = column_wise_mask[i-2] == num_replicates
        left_three = column_wise_mask[i-3] == num_replicates
        right_one = column_wise_mask[i+1] == num_replicates
        right_two = column_wise_mask[i+2] == num_replicates
        right_three = column_wise_mask[i+3] == num_replicates

        if np.sum([left_one, left_two, left_three, right_one, right_two, right_three]) >= 3:
            copied_mask[:,i] = True

             
    # second heuristic: selecting the largest continous interval
    row = copied_mask[0].tolist()
    all_length = []
    all_idxs = []
    tupled_all_idxs = []

    current_length = 0
    for i, r in enumerate(row):
        if r == True:    
            if current_length == 0:
                first_idx = i
                all_idxs.append(first_idx)
            current_length += 1    
        else:
            if current_length != 0:            
                all_length.append(current_length)
                last_idx = i
                all_idxs.append(last_idx)
            current_length = 0

    if current_length != 0:
        all_length.append(current_length)
        last_idx = i
        all_idxs.append(last_idx)

    for i in range(0, len(all_idxs), 2):
        tupled_all_idxs.append([all_idxs[i], all_idxs[i+1]])

    mask_idxs = tupled_all_idxs[np.argmax(all_length)]
    copied_mask = deepcopy(mask)
    copied_mask[:] = False
    copied_mask[:,mask_idxs[0]:mask_idxs[1]] = True
    
    return copied_mask




def compute_mask_from_tree_sequences(tree_sequences, population_time, num_cpus=1, num_replicates=100, min_coal_tree=50):

    num_time_windows = len(population_time)    
    binned_population_time = [[population_time[i], population_time[i+1]] for i in range(len(population_time)-1)]
    
    args = []
    for ts in tree_sequences:
        args.append((ts, binned_population_time, num_time_windows))

    with Pool(num_cpus) as p: 
        coalescent_times_replicates = p.starmap(get_binned_coalescent_times, tqdm(args, total=len(tree_sequences)))
    
    coalescent_times_replicates = np.array(coalescent_times_replicates)

    mask = coalescent_times_replicates >= min_coal_tree
    mask[:, mask.sum(0) >= min_coal_tree] = True
    mask[:, mask.sum(0) < min_coal_tree] = False
    mask = uniformize_mask_with_hacky_heuristic(mask, num_time_windows, num_replicates=num_replicates)
    
    return coalescent_times_replicates, mask


"""
def get_binned_coalescent_times(ts, binned_population_time, num_time_windows:int,  num_trees: int = 500) -> list:

        
    binned_population_time = np.array(binned_population_time)
    sorted_log_trees_node_times = get_sorted_log_trees_node_times(ts, discretize_trees(ts.aslist(), num_trees))
    tree_times = np.exp(np.array(sorted_log_trees_node_times))
        
    tree_bins = [] 
    outside_window = 0
    
    for current_tree_times in tree_times:
                
        for i, time in enumerate(current_tree_times):
            if time >= 100_000:
                outside_window += 1
                time_window = binned_population_time.shape[0]
                tree_bins.append(time_window)
            elif time < 1:
                outside_window += 1
                tree_bins.append(0)
            else:
                time_window = np.argwhere(np.sum(binned_population_time < time, axis=1) == 1).item()
                tree_bins.append(time_window)
            
    coalescent_times = sorted(np.array(tree_bins).flatten().tolist())
    coalescent_time_lookup = dict(Counter(coalescent_times))
    expand_coalescent_lookup(coalescent_time_lookup, num_time_windows=num_time_windows)
    coalescent_times = [item[1] for item in sorted(coalescent_time_lookup.items())]
    
    return coalescent_times
"""








def convert_tree_sequence_to_data_object_with_mask(tree_sequence: tskit.trees.TreeSequence,
                                                     parameter: np.ndarray,
                                                     mask: np.ndarray,
                                                     population_time: list,
                                                     num_trees:int = 500,
                                                     num_embedding:int = 50, 
                                                     directory: str = "datasets",
                           ):
    """ Converts tree_sequence to a data_object.
    
    
    Arg types:
        * **tree_sequence** *(tree sequence)* - A single tree sequence
        * **parameter** *(numpy array)* - Parameter for simulation.
        * **mask** *(tree sequence)* - A single tree sequence
        * **population_time** *(list of floats)* - Population times.
        * **num_trees** *(int)* - Number of trees used for discretization.
        * **num_embedding** *(int)* - emdbedding.
    
    """
    
    recombination_rate, mutation_rate, population_size, scenario, replicate = extract_parameter(parameter, pop_size_last_idx=len(population_time)+3)  
    
    y = torch.Tensor(population_size.tolist())
    #trees =  discretize_trees(tree_sequence.aslist(), num_trees = num_trees)
    trees = tree_sequence.aslist()[:num_trees]
    
    #mask = get_population_time_mask(get_sorted_log_trees_node_times(tree_sequence, trees), np.log(population_time), hard_lower_threshold=hard_lower_threshold)
    data_objects = datatize_trees(trees)

    max_num_nodes = 2 * tree_sequence.num_samples - 1 
    for _ , data in enumerate(data_objects):
        num_nodes = data.num_nodes
        data.x = torch.eye(max_num_nodes,num_embedding)
        data.x[num_nodes:] = torch.zeros(num_embedding)
        data.y = torch.Tensor(torch.log(y))
        data.num_nodes = max_num_nodes

    #return data_objects, mask
    torch.save((data_objects, mask), open(str(directory) + "data_" + str(scenario) + "_" + str(replicate) + ".pth", "wb"))



def convert_tree_sequences_to_data_objects_with_masks(tree_sequences: list[tskit.trees.TreeSequence],
                                           parameters: np.ndarray,
                                           masks: list[np.ndarray],
                                           population_time: np.ndarray,
                                           num_trees:int = 500,
                                           num_embedding:int = 50, 
                                           directory: str = "datasets",
                                           num_cpus: int = 1,
                           ):
    """ Converts tree_sequences to a data_objects.
    
    
    Arg types:
        * **tree_sequences** *(list of tree sequences)* - A list of tree sequences.
        * **parameter** *(numpy array)* - Parameter for simulation.
        * **population_time** *(numpy array)* - Population times.
        * **num_trees** *(int)* - Number of trees used for discretization.
        * **num_embedding** *(int)* - emdbedding.
        * **directory** *(str)* - Output directory.
    
    """
    if not os.path.exists(directory): os.makedirs(directory)

    
    args = []
    for ts, parameter, mask in zip(tree_sequences, np.array(parameters), masks):
        args.append((ts, parameter, mask,  population_time, num_trees, num_embedding, directory))

    with Pool(num_cpus) as p: 
        _ = p.starmap(convert_tree_sequence_to_data_object_with_mask, tqdm(args, total=len(tree_sequences)))


