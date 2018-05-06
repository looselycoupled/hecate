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

    def verify_dimensions(self):
        for item in self.buffer:
            if item["state"].shape != (84,84,4):
                import pdb; pdb.set_trace()
            if item["previous_state"].shape != (84,84,4):
                import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.buffer)
