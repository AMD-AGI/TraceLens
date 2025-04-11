import json

class DataLoader:
    @staticmethod
    def load_data(filename_path:str, save_preprocessed: bool = False) -> dict:
        if filename_path.endswith('pb'):
            from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
            data, _ = convert.xspace_to_tool_data([filename_path], "trace_viewer@^", {})
            data = data.decode("utf-8") # we get bytes back from the call above
            return json.loads(data)
        elif filename_path.endswith('json.gz'):
            import gzip
            with gzip.open(filename_path, 'r') as fin:
                data = fin.read().decode('utf-8')
            return json.loads(data)
        elif filename_path.endswith('json'):
            with open(filename_path, 'r') as fin:
                data = fin.read()
        else:
            raise ValueError("Unknown file type",filename_path)
        if (save_preprocessed):
            with open(filename_path.replace("pb", "processed.json"), 'w') as writefile:
                writefile.write(data)
        return json.loads(data)
