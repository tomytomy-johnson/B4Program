from collections import defaultdict
from common.utils import smoothing_history
from common.utils import plot_histories

class Emotion:
    def __init__(self):
        self.Joy = defaultdict(lambda: 0)
        self.Distress = defaultdict(lambda: 0)
        self.Hope = defaultdict(lambda: 0)
        self.Fear = defaultdict(lambda: 0)   
    