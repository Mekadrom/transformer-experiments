import os
import yaml

class YamlDict(dict):
    def __init__(self, *args, **kwargs):
        super(YamlDict, self).__init__(*args, **kwargs)
        
        self.__dict__ = self

    def __getattribute__(self, name):
        return self.__getitem__(name) if name in self else super().__getattribute__(name)

def load_yaml(file_path, ovr_args):
    file_path_dir = os.path.dirname(file_path)
    with open(os.path.join(file_path_dir, 'default.yaml'), 'r') as default_config:
        with open(file_path, 'r') as f:
            y: dict = yaml.safe_load(default_config)
            y.update(yaml.safe_load(f))
            if ovr_args is not None:
                y.update(ovr_args)
            return YamlDict(y)
