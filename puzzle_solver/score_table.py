import numpy as np

from .enums import JoinDirection


DIRECTIONS_BY_INDEX = tuple(JoinDirection)
DIRECTION_TO_INDEX = {
    direction: index
    for index, direction in enumerate(DIRECTIONS_BY_INDEX)
}


def createScoreTable(piece_count, score_storage="dense"):
    if score_storage == "dict":
        return {}
    if score_storage == "dense":
        return DenseScoreTable(piece_count)
    raise ValueError("score_storage must be 'dense' or 'dict'")


class DenseScoreTable:
    def __init__(self, piece_count):
        self.piece_count = piece_count
        shape = (piece_count + 1, len(DIRECTIONS_BY_INDEX), piece_count + 1)
        self._values = np.full(shape + (1,), np.nan, dtype=np.float64)
        self._filled = np.zeros(shape, dtype=bool)
        self._scalar = np.ones(shape, dtype=bool)
        self._extra = {}

    def _scoreIndices(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            return None
        own_number, direction, join_number = key
        if direction not in DIRECTION_TO_INDEX:
            return None
        if not isinstance(own_number, int) or not isinstance(join_number, int):
            return None
        if not 1 <= own_number <= self.piece_count:
            return None
        if not 1 <= join_number <= self.piece_count:
            return None
        return own_number, DIRECTION_TO_INDEX[direction], join_number

    def _ensureComponentCount(self, component_count):
        current_count = self._values.shape[-1]
        if component_count <= current_count:
            return
        values = np.full(
            self._values.shape[:-1] + (component_count,),
            np.nan,
            dtype=np.float64,
        )
        values[..., :current_count] = self._values
        self._values = values

    def _scoreComponents(self, score):
        if isinstance(score, tuple):
            return tuple(float(value) for value in score), False
        return (float(score),), True

    def __setitem__(self, key, score):
        indices = self._scoreIndices(key)
        if indices is None:
            self._extra[key] = score
            return
        self._setScore(indices, score)

    def _setScore(self, indices, score):
        components, is_scalar = self._scoreComponents(score)
        self._ensureComponentCount(len(components))
        self._values[indices + (slice(0, len(components)),)] = components
        if len(components) < self._values.shape[-1]:
            self._values[indices + (slice(len(components), None),)] = np.nan
        self._filled[indices] = True
        self._scalar[indices] = is_scalar

    def __getitem__(self, key):
        indices = self._scoreIndices(key)
        if indices is None:
            return self._extra[key]
        if not self._filled[indices]:
            raise KeyError(key)
        if self._scalar[indices]:
            return float(self._values[indices + (0,)])
        component_count = self._values.shape[-1]
        return tuple(float(value) for value in self._values[indices][:component_count])

    def __contains__(self, key):
        indices = self._scoreIndices(key)
        if indices is None:
            return key in self._extra
        return bool(self._filled[indices])

    def __iter__(self):
        yield from self._extra
        for own_number, direction_index, join_number in np.argwhere(self._filled):
            yield (
                int(own_number),
                DIRECTIONS_BY_INDEX[int(direction_index)],
                int(join_number),
            )

    def __len__(self):
        return int(np.count_nonzero(self._filled)) + len(self._extra)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def values(self):
        for key in self:
            yield self[key]

    def items(self):
        for key in self:
            yield key, self[key]

    def update(self, values):
        for key, score in values.items():
            self[key] = score

    def setMany(self, entries):
        entries = list(entries)
        if not entries:
            return

        entry_count = len(entries)
        first_score = entries[0][1]
        first_is_scalar = not isinstance(first_score, tuple)
        component_count = 1 if first_is_scalar else len(first_score)

        own_numbers = np.empty(entry_count, dtype=np.intp)
        direction_indices = np.empty(entry_count, dtype=np.intp)
        join_numbers = np.empty(entry_count, dtype=np.intp)
        flat_indices = np.empty(entry_count, dtype=np.intp)
        scores = np.empty((entry_count, component_count), dtype=np.float64)
        own_stride = len(DIRECTIONS_BY_INDEX) * (self.piece_count + 1)
        direction_stride = self.piece_count + 1

        for entry_index, (key, score) in enumerate(entries):
            indices = self._scoreIndices(key)
            if indices is None:
                self._setManyOneByOne(entries)
                return

            is_scalar = not isinstance(score, tuple)
            if is_scalar != first_is_scalar:
                self._setManyOneByOne(entries)
                return
            if is_scalar:
                scores[entry_index, 0] = score
            elif len(score) == component_count:
                scores[entry_index] = score
            else:
                self._setManyOneByOne(entries)
                return

            own_number, direction_index, join_number = indices
            own_numbers[entry_index] = own_number
            direction_indices[entry_index] = direction_index
            join_numbers[entry_index] = join_number
            flat_indices[entry_index] = (
                own_number * own_stride
                + direction_index * direction_stride
                + join_number
            )

        if np.unique(flat_indices).size != entry_count:
            self._setManyOneByOne(entries)
            return

        self._ensureComponentCount(component_count)
        self._values[
            own_numbers,
            direction_indices,
            join_numbers,
            :component_count,
        ] = scores
        if component_count < self._values.shape[-1]:
            self._values[
                own_numbers,
                direction_indices,
                join_numbers,
                component_count:,
            ] = np.nan
        self._filled[own_numbers, direction_indices, join_numbers] = True
        self._scalar[own_numbers, direction_indices, join_numbers] = first_is_scalar

    def _setManyOneByOne(self, entries):
        for key, score in entries:
            self[key] = score

    def setManyArrays(
            self,
            own_numbers,
            direction_indices,
            join_numbers,
            scores,
            is_scalar=True):
        if len(own_numbers) == 0:
            return

        own_numbers = np.asarray(own_numbers, dtype=np.intp)
        direction_indices = np.asarray(direction_indices, dtype=np.intp)
        join_numbers = np.asarray(join_numbers, dtype=np.intp)
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores[:, np.newaxis]

        component_count = scores.shape[1]
        self._ensureComponentCount(component_count)
        self._values[
            own_numbers,
            direction_indices,
            join_numbers,
            :component_count,
        ] = scores
        if component_count < self._values.shape[-1]:
            self._values[
                own_numbers,
                direction_indices,
                join_numbers,
                component_count:,
            ] = np.nan
        self._filled[own_numbers, direction_indices, join_numbers] = True
        self._scalar[own_numbers, direction_indices, join_numbers] = is_scalar

    def normalizeCombinedScores(self):
        if self._values.shape[-1] < 2:
            return
        mask = self._filled
        if not np.any(mask):
            return

        mahalanobis_values = self._values[..., 0][mask]
        euclidean_values = self._values[..., 1][mask]
        max_mahalanobis = mahalanobis_values.max()
        min_mahalanobis = mahalanobis_values.min()
        max_euclidean = euclidean_values.max()
        min_euclidean = euclidean_values.min()
        if max_mahalanobis == min_mahalanobis or max_euclidean == min_euclidean:
            raise ZeroDivisionError("score range is zero")

        normalized_scores = (
            (mahalanobis_values - min_mahalanobis)
            / (max_mahalanobis - min_mahalanobis)
            + (euclidean_values - min_euclidean)
            / (max_euclidean - min_euclidean)
        )
        score_values = self._values[..., 0]
        score_values[mask] = normalized_scores
        for component_index in range(1, self._values.shape[-1]):
            component_values = self._values[..., component_index]
            component_values[mask] = np.nan
        self._scalar[mask] = True

    def applyReliabilityScores(self):
        score_values = self._values[..., 0]
        for own_number in range(1, self.piece_count + 1):
            for direction_index in range(len(DIRECTIONS_BY_INDEX)):
                mask = self._filled[own_number, direction_index, :]
                if not np.any(mask):
                    continue
                raw_scores = score_values[own_number, direction_index, mask]
                if len(raw_scores) < 2:
                    second_best_score = raw_scores[0]
                else:
                    second_best_score = np.partition(raw_scores, 1)[1]
                updated_scores = self._reliabilityScores(
                    raw_scores,
                    second_best_score,
                )
                score_values[own_number, direction_index, mask] = updated_scores
                self._scalar[own_number, direction_index, mask] = True

    def _reliabilityScores(self, scores, second_best_score):
        if second_best_score <= 0:
            return np.where(scores <= 0, 1.0, np.inf)
        return scores / second_best_score

    def asDict(self):
        return {
            key: score
            for key, score in self.items()
        }
