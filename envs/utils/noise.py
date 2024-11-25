"""
    Noise modules 
"""
import numpy as np

class Noise:
    """Base class for noise generation."""
    def __init__(self, amplitude=1.0):
        self.amplitude = amplitude
    
    def generate(self, size):
        """Generate noise with specified size."""
        raise NotImplementedError("Subclasses should implement this!")

class WhiteNoise(Noise):
    """Class to generate white noise."""
    def generate(self, size):
        """Generate white noise with given size."""
        return self.amplitude * np.random.normal(0, 1, size)

class PinkNoise(Noise):
    """Class to generate pink noise (1/f noise) using the Voss-McCartney algorithm."""
    def __init__(self, amplitude=1.0, num_sources=16):
        super().__init__(amplitude)
        self.num_sources = num_sources
        self.state = np.zeros(num_sources)
    
    def generate(self, size):
        """Generate pink noise with given size."""
        output = np.zeros(size)
        for i in range(size):
            idx = np.random.randint(0, self.num_sources)
            self.state[idx] = np.random.normal(0, 1)
            output[i] = np.sum(self.state)
        return self.amplitude * output / self.num_sources

# Example usage with sigma_w and sigma_p from Allan deviation results
if __name__ == "__main__":
    # Assume these values were obtained from the Allan deviation analysis
    sigma_w = 0.01  # White noise coefficient
    sigma_p = 0.005  # Pink noise coefficient

    white_noise = WhiteNoise(amplitude=sigma_w)
    pink_noise = PinkNoise(amplitude=sigma_p)

    # Generate 1000 samples of white noise and pink noise
    white_samples = white_noise.generate(1000)
    pink_samples = pink_noise.generate(1000)
