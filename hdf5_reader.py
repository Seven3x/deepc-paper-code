import h5py
import os
import numpy as np

class HDF5Reader:
    def __init__(self, directory=None):
        self.directory = directory

    def read_hdf5_file(self, file_path):
        with h5py.File(file_path, 'r') as file:
            # Load the data (assuming it's a dictionary-like structure)
            data = {key: file[key][()] for key in file.keys()}
        return data

    def list_files_in_directory(self):
        return [f for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f))]

    def choose_file(self, files, data_type):
        print("-------------------------------------")
        for i, file in enumerate(files):
            print(f"{i + 1}: {file}")
        choice = int(input(f"{data_type} - Enter the number of the file you want to choose: ")) - 1
        print("-------------------------------------")
        return files[choice]

    def run(self, data_type):
        files = self.list_files_in_directory()
        chosen_file = self.choose_file(files, data_type)
        file_path = os.path.join(self.directory, chosen_file)
        data = self.read_hdf5_file(file_path)
        chosen_file = chosen_file.strip(".hdf5")
        return data, chosen_file
    
    def save_to_hdf5(self, data, filename):
        with h5py.File(filename, 'w') as f:
            for key, value in data.items():
                # Check if the value is a numpy ndarray
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                else:
                    # Save other types of data as attributes (strings, integers, etc.)
                    f.attrs[key] = value