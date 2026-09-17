"""Fixed total-capacity niches using CodeBLEU syntax/dataflow landmarks."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def python_language():
    """Bridge CodeBLEU 0.7's tree-sitter 0.22 to Python grammar capsules.

tree-sitter-python 0.23 ships ARM macOS wheels but returns a PyCapsule;
tree-sitter 0.22 expects the underlying TSLanguage pointer. No global patching.
"""
    import ctypes
    from tree_sitter import Language
    import tree_sitter_python
    grammar = tree_sitter_python.language()
    try:
        return Language(grammar)
    except TypeError:
        pointer = ctypes.pythonapi.PyCapsule_GetPointer
        pointer.restype = ctypes.c_void_p
        pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        return Language(pointer(grammar, b'tree_sitter.Language'))


@dataclass
class Candidate:
    id: int
    code: str
    concept: str
    result: dict
    niche: int = 0

    @property
    def score(self):
        return self.result['score']

    @property
    def digest(self):
        return hashlib.sha256(self.code.encode()).hexdigest()


class NichePopulation:
    def __init__(self, capacity=16, partitions=4, seed=2024):
        if not 1 <= partitions <= capacity:
            raise ValueError('require 1 <= partitions <= total population capacity')
        self.capacity, self.partitions, self.seed = capacity, partitions, seed
        self.members = []
        self.best = None
        self.anchors = []
        self.scaler = self.pca = self.kmeans = None
        self.active_partitions = 1

    @staticmethod
    @lru_cache(maxsize=1024)
    def similarity(a, b):
        from codebleu.syntax_match import corpus_syntax_match
        from codebleu.dataflow_match import corpus_dataflow_match
        # CodeBLEU matching can be asymmetric: average both orientations.
        values = []
        for left, right in ((a, b), (b, a)):
            language = python_language()
            syntax = corpus_syntax_match([[left]], [right], 'python', tree_sitter_language=language)
            dataflow = corpus_dataflow_match([[left]], [right], 'python', tree_sitter_language=language)
            values.append(0.5 * syntax + 0.5 * dataflow)
        return float(np.mean(values))

    def features(self, code):
        return [self.similarity(code, anchor) for anchor in self.anchors]

    def _initialize(self):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.anchors = [x.code for x in self.members]
        raw = np.asarray([self.features(x.code) for x in self.members])
        self.scaler = StandardScaler().fit(raw)
        scaled = self.scaler.transform(raw)
        if not np.any(np.var(scaled, axis=0) > 1e-12):
            # Truly indistinguishable features: keep a single niche explicitly.
            self.active_partitions = 1
            return
        self.pca = PCA(n_components=min(10, *scaled.shape)).fit(scaled)
        features = self.pca.transform(scaled)
        count = min(self.partitions, len(np.unique(np.round(features, 10), axis=0)))
        self.kmeans = KMeans(n_clusters=count, random_state=self.seed, n_init=10).fit(features)
        self.active_partitions = count
        for x, label in zip(self.members, self.kmeans.labels_):
            x.niche = int(label)

    def register(self, child):
        if not isinstance(child.score, (int, float)) or not np.isfinite(child.score):
            return False
        if self.best is None or child.score > self.best.score:
            self.best = child
        existing = next((x for x in self.members if x.digest == child.digest), None)
        if existing:
            if child.score <= existing.score:
                return False
            self.members.remove(existing)
        if self.kmeans is not None:
            feature = self.pca.transform(self.scaler.transform([self.features(child.code)]))
            child.niche = int(self.kmeans.predict(feature)[0])
        self.members.append(child)
        if self.partitions > 1 and not self.anchors and len(self.members) >= self.partitions:
            self._initialize()
        # Total storage is fixed across partition-count ablations.
        quotas = [self.capacity // self.active_partitions + (i < self.capacity % self.active_partitions)
                  for i in range(self.active_partitions)]
        self.members = [x for i, q in enumerate(quotas)
                        for x in sorted((x for x in self.members if x.niche == i),
                                        key=lambda x: x.score, reverse=True)[:q]]
        return any(x.id == child.id for x in self.members)

    def select(self, rng, cross=False):
        if not self.members:
            return []
        niches = sorted({x.niche for x in self.members})
        niche = rng.choice(niches)
        local = [x for x in self.members if x.niche == niche]
        # Cold start expands from random parents, then local tournament selection.
        parent = rng.choice(local) if not self.anchors and self.partitions > 1 else max(
            rng.sample(local, min(3, len(local))), key=lambda x: x.score)
        parents = [parent]
        if cross:
            others = [x for x in self.members if x.niche != niche]
            if not others:
                others = [x for x in self.members if x.id != parent.id]
            if others:
                parents.append(rng.choice(others))
        return parents
