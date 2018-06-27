import numpy as np
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, mu, theta, sigma):
        """Initialize OUNoise instance."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        """Reset state value to default."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu-x) * np.random.randn(len(x))
        self.state = x+dx
        return self.state
