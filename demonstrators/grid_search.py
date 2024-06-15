import random
import numpy as np
from copy import deepcopy
from itertools import product
from joblib import Parallel, delayed


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


class ComponentA:
    def __init__(self, param_a=None, param_b=None):
        self.param_a = param_a
        self.param_b = param_b

    def __repr__(self):
        return f"ComponentA(param_a={self.param_a}, param_b={self.param_b})"


class ComponentB:
    def __init__(self, param_c, param_d):
        self.param_c = param_c
        self.param_d = param_d

    def __repr__(self):
        return f"ComponentB(param_c={self.param_c}, param_d={self.param_d})"


class Pipeline:
    def __init__(self, components: list) -> None:
        self.components = components

    def add_component(self, component):
        self.components.append(component)

    def __repr__(self) -> str:
        string = "Pipeline\n"
        string += "\n".join([f"\t {component}" for component in self.components])
        string += "\n"
        return string

    def run(self):
        print(".", end="")


def score_f1(pipeline: Pipeline, expected_concepts, expected_relations):
    # simulate the get ratio file
    pipeline.run()
    return random.random()


class GridSearch:
    def __init__(
        self,
        components_params_grid: dict,
        base_pipeline: Pipeline = Pipeline([]),
        scoring=score_f1,
        n_jobs=None,
    ):
        if isinstance(components_params_grid, dict):
            components_params_grid = [components_params_grid]
        self.components_params_grid = components_params_grid  # grille des composnants
        self.base_pipeline = base_pipeline  # pipeline de base
        self.pipelines = []
        self.n_jobs = n_jobs  # nbre de processus lors de l'apprentissage
        self.scoring = scoring  # fonction de scroring on travaillera avec la f1
        self.best_pipeline = None  # resultat du gride search

    def _create_pipelines(self):

        for grid in self.components_params_grid:
            items = grid.items()
            if not items:
                self.pipelines = []
                print("no params")
            else:
                components_instances = {}
                for component, param_grid in items:
                    params = [param for param in ParameterGrid(param_grid)]
                    components_object = [component(**param) for param in params]
                    components_instances.update({component: components_object})

                _, components_object = zip(*components_instances.items())

                # à Adpater pour les pipelines d'olaf
                for pipeline_components in product(*components_object):
                    pipeline = deepcopy(self.base_pipeline)
                    for pipeline_component in pipeline_components:
                        pipeline.add_component(pipeline_component)
                    self.pipelines.append(pipeline)

    def fit_pipeline(self, expected_concepts, expected_relations):
        self._create_pipelines()
        parallel = Parallel(n_jobs=self.n_jobs)
        print(self.pipelines)
        results = parallel(
            delayed(self.scoring)(pipeline, expected_concepts, expected_relations)
            for pipeline in self.pipelines
        )

        self.best_pipeline, self.best_score = (
            self.pipelines[np.argmax(results)],
            results[np.argmax(results)],
        )


if __name__ == "__main__":
    component_grid = {
        ComponentA: {"param_a": [2], "param_b": [3, 8]},
        ComponentB: {"param_c": [6, 7], "param_d": [9]},
    }

    grid_search = GridSearch(
        components_params_grid=component_grid, scoring=score_f1, n_jobs=3
    )
    grid_search.fit_pipeline(expected_concepts=["hello"], expected_relations=["world"])
    print(grid_search.pipelines)
    print(grid_search.best_score)
    print(grid_search.best_pipeline)
