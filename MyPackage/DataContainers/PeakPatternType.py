from enum import Enum


class PatternDirection(Enum):
    UP = 1
    DOWN = 2


class PeakPatternType:
    def __init__(self, peaks_count: float, direction: PatternDirection):
        self.peaks_count = peaks_count
        self.direction = direction

    def __str__(self):
        return str(self.peaks_count) + ('$\\uparrow$' if self.direction == PatternDirection.UP else '$\\downarrow$')

    def __repr__(self):
        return f'{self.peaks_count} {self.direction}'

    def __hash__(self):
        return int(round(self.peaks_count*10)) + (1000 if self.direction == PatternDirection.UP else 2000)

    def __eq__(self, other):
        return isinstance(other, PeakPatternType) \
               and other.peaks_count == self.peaks_count \
               and other.direction == self.direction

    def __lt__(self, other):
        assert isinstance(other, type(self))
        if self.peaks_count != other.peaks_count:
            return self.peaks_count < other.peaks_count
        else:
            return self.direction == PatternDirection.UP and other.direction == PatternDirection.DOWN
