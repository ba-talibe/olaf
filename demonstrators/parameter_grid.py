from itertools import product


class ParameterGrid:
    def __init__(self, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        for grid in self.param_grid:
            items = sorted(grid.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        return sum(len(list(product(*grid.values()))) for grid in self.param_grid)

    def __getitem__(self, index):
        for grid in self.param_grid:
            items = sorted(grid.items())
            if not items:
                if index == 0:
                    return {}
                index -= 1
                continue
            keys, values = zip(*items)
            size = len(list(product(*values)))
            if index < size:
                return dict(zip(keys, list(product(*values))[index]))
            index -= size
        raise IndexError("ParameterGrid index out of range")


if __name__ == "__main__":
    params_grid = [{"param1": [1, 2], "param2": [True, False]}]
    params = ParameterGrid(param_grid=params_grid)
    params = [param for param in params]
    print(params)
