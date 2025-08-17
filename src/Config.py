import yaml
import warnings

class Config:
    def __init__(self, config_path=None):
        self._config_data = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path):
        with open(config_path, 'r') as file:
            raw_data = yaml.safe_load(file)
        for key, value in raw_data.items():
            setattr(self, key, value)

    def save(self, output_path):
        with open(output_path, 'w') as file:
            yaml.dump(self._config_data, file)

    def __getattr__(self, name):
        if name in self._config_data:
            return self._config_data[name]
        warnings.warn(f"'Config' object has no attribute '{name}'", UserWarning)
        return None

    def __setattr__(self, name, value):
        if name in ['_config_data']:
            super().__setattr__(name, value)
        else:
            self._config_data[name] = value
            super().__setattr__(name, value)
    
    def __iter__(self):
        return iter(self._config_data.items())
    
    def __repr__(self):
        pass