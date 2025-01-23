# callable_dict.py

class CallableDict(dict):
    """
    A dictionary that is also callable. Allows access via indexing and calling.
    Example:
        d = CallableDict({'a': 1, 'b': 2})
        print(d['a'])  # Outputs: 1
        print(d('b'))  # Outputs: 2
    """
    def __call__(self, key):
        return self[key]
