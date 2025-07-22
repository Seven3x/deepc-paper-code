import json
import os

class JsonReader:
    def __init__(self, results_directory=None):
        self.results_directory = results_directory

    def read_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def write_json_file(self, filename, data):
        """Writes the given dictionary to a JSON file in the results directory."""
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)  # Pretty-print for readability

    def list_files_in_directory(self):
        return [f for f in os.listdir(self.results_directory) if os.path.isfile(os.path.join(self.results_directory, f))]

    def choose_file(self, files, data_type):
        for i, file in enumerate(files):
            print(f"{i + 1}: {file}")
        choice = int(input(f"{data_type} - Enter the number of the file you want to choose: ")) - 1
        return files[choice]

    def run(self, data_type):
        files = self.list_files_in_directory()
        chosen_file = self.choose_file(files, data_type)
        file_path = os.path.join(self.results_directory, chosen_file)
        data = self.read_json_file(file_path)
        chosen_file = chosen_file.strip(".json")
        return data, chosen_file