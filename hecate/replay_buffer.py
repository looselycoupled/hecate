import random

class ReplayBuffer(object):

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def trim(self):
        self.buffer = self.buffer[-self.size:]

    def append(self, record):
        self.buffer.append(record)
        if len(self.buffer) > self.size:
            self.trim()

    def sample(self, amount=32):
        return random.sample(self.buffer, amount)

    def __len__(self):
        return len(self.buffer)
