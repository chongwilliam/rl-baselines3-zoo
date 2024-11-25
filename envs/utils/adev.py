"""
    Allan deviation characterization 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class AllanDeviation:
    def __init__(self, data, sample_rate):
        """
        Initialize the AllanDeviation class.
        
        Parameters:
        - data: 1D array of sensor measurements.
        - sample_rate: Sampling rate of the data in Hz.
        """
        self.data = data
        self.sample_rate = sample_rate

    def compute_allan_deviation(self, max_cluster_size=None):
        """
        Compute the Allan deviation for the data.
        
        Parameters:
        - max_cluster_size: Maximum number of points to cluster. Default is half the data length.
        
        Returns:
        - tau: Array of cluster times (integration times).
        - adev: Array of Allan deviation values.
        """
        n = len(self.data)
        if max_cluster_size is None:
            max_cluster_size = n // 2

        cluster_sizes = np.logspace(0, np.log10(max_cluster_size), num=50, dtype=int)
        cluster_sizes = np.unique(cluster_sizes)

        tau = cluster_sizes / self.sample_rate
        adev = np.zeros(len(cluster_sizes))

        for i, m in enumerate(cluster_sizes):
            cluster_averages = np.add.reduceat(self.data, np.arange(0, n, m)) / m
            diff = np.diff(cluster_averages)
            adev[i] = np.sqrt(0.5 * np.mean(diff**2))

        return tau, adev

    def fit_white_noise(self, tau, adev):
        """
        Estimate the white noise coefficient by fitting the first portion of the Allan deviation curve.
        
        Parameters:
        - tau: Array of cluster times.
        - adev: Array of Allan deviation values.
        
        Returns:
        - White noise coefficient.
        """
        # Fit only the initial portion with a slope of -0.5
        def white_noise_model(tau, sigma_w):
            return sigma_w / np.sqrt(tau)

        popt, _ = curve_fit(white_noise_model, tau, adev, maxfev=10000)
        return popt[0]

    def fit_pink_noise(self, tau, adev):
        """
        Estimate the pink noise coefficient by fitting the flat portion of the Allan deviation curve.
        
        Parameters:
        - tau: Array of cluster times.
        - adev: Array of Allan deviation values.
        
        Returns:
        - Pink noise coefficient.
        """
        # Fit only the flat portion (constant pink noise level)
        def pink_noise_model(tau, sigma_p):
            return sigma_p * np.ones_like(tau)

        popt, _ = curve_fit(pink_noise_model, tau, adev, maxfev=10000)
        return popt[0]

    def plot_allan_deviation(self, tau, adev, sigma_w=None, sigma_p=None):
        """
        Plot the Allan deviation curve with optional noise parameters.
        
        Parameters:
        - tau: Array of cluster times.
        - adev: Array of Allan deviation values.
        - sigma_w: Estimated white noise coefficient (optional).
        - sigma_p: Estimated pink noise coefficient (optional).
        """
        plt.loglog(tau, adev, label='Allan Deviation')
        if sigma_w is not None:
            plt.loglog(tau, sigma_w / np.sqrt(tau), 'r--', label=f'White Noise: {sigma_w:.2e}')
        if sigma_p is not None:
            plt.loglog(tau, sigma_p * np.ones_like(tau), 'g--', label=f'Pink Noise: {sigma_p:.2e}')
        plt.xlabel("Cluster Time (s)")
        plt.ylabel("Allan Deviation")
        plt.title("Allan Deviation Curve")
        plt.grid(which="both", linestyle="--")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data (e.g., simulated sensor noise)
    sample_rate = 100  # Hz
    duration = 100  # seconds
    np.random.seed(0)
    data = np.cumsum(np.random.normal(0, 1, sample_rate * duration))  # Random walk

    allan_dev = AllanDeviation(data, sample_rate)
    tau, adev = allan_dev.compute_allan_deviation()
    
    # Estimate white and pink noise coefficients
    sigma_w = allan_dev.fit_white_noise(tau, adev)
    sigma_p = allan_dev.fit_pink_noise(tau, adev)

    # Plot the Allan deviation curve with noise estimates
    allan_dev.plot_allan_deviation(tau, adev, sigma_w=sigma_w, sigma_p=sigma_p)
