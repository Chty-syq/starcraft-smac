import random
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.capacity = args.buffer_size
        self.memory = []
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        with self.lock:
            self.memory.append(transition)
            if len(self.memory) > self.capacity:
                del self.memory[0]

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)
