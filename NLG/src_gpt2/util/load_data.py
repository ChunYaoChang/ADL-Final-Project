import json 

def load_json_file(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
        return data 

def load_data(data_dir):
    data_list = []
    for file_path in data_dir.iterdir():
        data_list.append(load_json_file(file_path))
    # Flatten data_list 
    return [dialogue for per_data_list in data_list for dialogue in per_data_list]