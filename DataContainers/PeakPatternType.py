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
        return self.peaks_count.__hash__() ^ self.direction.__hash__()

    def __eq__(self, other):
        return isinstance(other, PeakPatternType) \
               and other.peaks_count == self.peaks_count \
               and other.direction == self.direction
