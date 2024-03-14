
import pickle
import torch

#####################################################
#####################################################

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

#####################################################
#####################################################

def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

#####################################################
#####################################################

def save_model(model, model_dir, ii):
    torch.save(model, model_dir+f'\\model_{ii}.pt')