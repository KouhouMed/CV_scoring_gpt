import yaml
import os


class ScorerConfig:
    def __init__(self, config_path='./scorer_config.yml'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            return self.get_default_config()

        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_default_config(self):
        return {
            'global': {
                'experience_weight': 0.4,
                'skill_weight': 0.3,
                'certification_weight': 0.2,
                'project_weight': 0.1
            },
            'skill_scoring': {
                'experience_weight': 0.4,
                'skill_weight': 0.3,
                'certification_weight': 0.2,
                'project_weight': 0.1
            },
            'models': {
                'chatgpt': {
                    'temperature': 0.2,
                    'max_tokens': 500
                }
            }
        }

    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k not in value:
                return default
            value = value[k]
        return value

    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save_config()
