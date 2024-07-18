import h5py
import matplotlib.pyplot as plt
import math

filename = "cache_valeriot/64/mih_lsh_0_10000_1B_R0.h5"

knn = int(input("Insert the number of K-NN to plot"))

if(knn % 10 != 0): exit

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array

    times_1 = []
    db_sizes_1 = []
    times_2 = []
    db_sizes_2 = []
    times_3 = []
    db_sizes_3= []

    curr_size = -1;

    for element in ds_arr:
        if(element[2] == knn and curr_size < element[0]):
            db_sizes_1.append((element[0] / 10e6))
            times_1.append(element[7])
            curr_size = element[0]

    plt.figure(figsize=(12, 6))
    plt.plot(db_sizes_1, times_1, marker='o', label=str(knn) + '-NN', color='red')
    plt.ylabel('times per query (s)')
    plt.xlabel('dataset size (milions)')
    plt.legend()
    plt.grid(True)
    plt.show()
