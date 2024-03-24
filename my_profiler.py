import numpy as np
import time


class Profiler:
    def __init__(self):
        self.names = []
        self.times = {}
        self.s_times = {}

    def tick(self, name):
        if name not in self.names:
            self.times[name] = []
            self.names.append(name)
        assert name not in self.s_times
        self.s_times[name] = time.time()

    def tock(self, name):
        d_time = time.time() - self.s_times[name]
        self.s_times.pop(name)
        self.times[name].append(d_time)

    def report(self):
        avg_time = {}
        for key in self.names:
            avg_time[key] = np.mean(self.times[key]).item()
        return avg_time
