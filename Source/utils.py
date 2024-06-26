import pickle  

def save_file(name, obj):
    """
    Function to save an object as a pickle file.
    
    Parameters:
    name (str): The file name to save the object to.
    obj: The object to be saved.

    Returns:
    None
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def load_file(name):
    """
    Function to load a pickle object from a file.

    Parameters:
    name (str): The name of the pickle file to load.

    Returns:
    obj: The loaded object.
    """
    return pickle.load(open(name, "rb"))
