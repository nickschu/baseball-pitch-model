import pickle

def get_pck_key_size(pd_file):
    with open(pd_file, 'rb') as f:
        d = pickle.load(f)
    return len(d.keys())

def get_pck_value_size(pd_file):
    with open(pd_file, 'rb') as f:
        d = pickle.load(f)
    return len(set(d.values()))