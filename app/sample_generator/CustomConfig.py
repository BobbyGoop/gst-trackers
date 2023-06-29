import json


class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CustomConfig(object):
    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return CustomConfig.load_dict(data)
        elif type(data) is list:
            return CustomConfig.load_list(data)
        else:
            return data

    @staticmethod
    def load_dict(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = CustomConfig.__load__(value)
        return result

    @staticmethod
    def load_list(data: list):
        result = [CustomConfig.__load__(item) for item in data]
        return result

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as f:
            result = CustomConfig.__load__(json.loads(f.read()))
        return result
