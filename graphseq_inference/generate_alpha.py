import numpy as np
import random
import torch

seed = 1234567890

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

import sys
from sklearn.linear_model import LinearRegression

computer="anon/projects2"
#computer = "ubuntu"

sys.path.append("/home/" + str(computer) + "/graphseq-inference/")
#sys.path.append("/home/" + str(computer) + "/graphseq-inference/graphseq_inference/")

print(sys.path)

data_dir = "/home/" + str(computer) + "/graphseq-inference-analysis/"


from graphseq_inference.data_utils import *


torch.Tensor(0)


#population_time = get_population_time(time_rate=0.1, num_time_windows=60, tmax=10_000_000).tolist()

from scipy.interpolate import interp1d

num_replicates = 100
upper_tree_limit = 500
num_scenario = 2200


def get_sequence_length(alpha):
    X = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]).reshape(-1, 1)
    y = np.array([903272667.6483952, 449680355.2510645,178743721.59285852, 70857015.72050871,26609043.76117958,
                  10475248.838195799,4518011.131732858, 2024414.914914666, 1265514.8452846785])
    y = y * 1.1
    y = np.log(y)
    reg = LinearRegression().fit(X, y)

    if alpha >= 1.9:
        alpha = 1.9
    
    sequence_length = int(np.exp(reg.predict(np.array([[alpha]])).item()))
    
    return sequence_length



for increment in range(0, num_scenario):


    upper_out_of_bound = lower_out_of_bound = True
    while upper_out_of_bound or lower_out_of_bound:
        steps = 18
        x = np.log(get_population_time(time_rate=0.1, num_time_windows=steps, tmax=10_000_000).tolist())
        y = np.log(sample_population_size(10_000, 10_000_000, steps))
        xnew = np.linspace(x[0], x[-1], num=10000, endpoint=True)
        f_cubic = interp1d(x, y, kind='cubic')
        ynew = f_cubic(xnew)
        upper_out_of_bound = np.sum(np.exp(ynew) > 10_000_000) > 0
        lower_out_of_bound = np.sum(np.exp(ynew) < 10_000) > 0
        x_sample = xnew[np.linspace(10, 9999, 60).astype(int)]
        y_sample = ynew[np.linspace(10, 9999, 60).astype(int)]
        population_time = np.exp(x_sample)
        population_size = np.exp(y_sample)





    random_alpha = np.round(np.random.uniform(1.01, 1.99), 2)
    #sequence_length = int(np.exp(reg.predict(np.array([[random_alpha]])).item()))
    sequence_length = get_sequence_length(random_alpha)
    print(f"alpha {random_alpha} sequence length {sequence_length}")

    parameters = sample_parameters(1,
                                   increment=increment,
                                   num_replicates=num_replicates,
                                   num_time_windows=60,
                                   n_min = 10_000,
                                   n_max = 10_000_000,
                                   recombination_rates=[1e-8, 1e-8],
                                   population_size=population_size,
                                   model="beta",
                                   alpha=random_alpha,
                                  )

    tree_sequences = simulate_tree_sequence(parameters,
                                            population_time,
                                            segment_length=sequence_length,
                                            num_replicates=num_replicates)
    
    num_trees = []
    for ts in tree_sequences:
        #if ts.num_trees >= 500:
        num_trees.append(ts.num_trees)

    print(random_alpha, int(np.mean(num_trees))) 
    print(np.sum(num_trees)/50000)


    while np.sum(num_trees)/50000 < 1.0:


        print("increasing sequence length until at least 500 trees", sequence_length)

        sequence_length = sequence_length * 1.2

        tree_sequences = simulate_tree_sequence(parameters,
                                            population_time,
                                            segment_length=sequence_length,
                                            num_replicates=num_replicates)


        num_trees = []
        for ts in tree_sequences:
            #if ts.num_trees >= 500:
            num_trees.append(ts.num_trees)






    parameters.to_csv(data_dir + "alpha-dataset/parameters_" + str(increment) + ".csv")

#    try:
    col_events , mask = compute_mask_from_tree_sequences(tree_sequences, population_time, num_cpus=7, min_coal_tree=30)
    convert_tree_sequences_to_data_objects_with_masks(tree_sequences,
                                                          parameters,
                                                          mask,
                                                          population_time, 
                                                          num_embedding=60,
                                                          num_trees=upper_tree_limit,
                                                          directory= data_dir + "alpha-dataset/mmc_dataset_" + str(increment) + "/",
                                                          num_cpus=20)

    #except Exception as e: print(e)
