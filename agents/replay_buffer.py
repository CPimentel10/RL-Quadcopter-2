from collections import deque, namedtuple
import random


class ReplayBuffer:
    """Fixed size buffer to store experience tuples."""

    def __init__(self, mem_size, batch_size):
        """Initialize a ReplayBuffer instance."""
        self.memory = deque(maxlen=mem_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                      'state', 'action', 'reward',
                                      'next_state', 'done'])

    def memorize(self, state, action, reward, next_state, done):
        """Add an experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Generate and return a random sample from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return length of ReplayBuffer."""
        return len(self.memory)
