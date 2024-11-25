from scipy.signal import butter

class LowPassFilter:
    def __init__(self, cutoff, fs, order=4):
        """
        Initializes the Low-Pass Filter.
        
        Parameters:
            cutoff (float): The cutoff frequency of the filter in Hz.
            fs (float): The sampling frequency in Hz.
            order (int): The order of the filter.
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        
        # Normalized cutoff frequency
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        
        # Calculate filter coefficients
        self.b, self.a = butter(self.order, normalized_cutoff, btype='low', analog=False)
        
        # Initialize previous input and output values (for iterative filtering)
        self.prev_input = 0.0
        self.prev_output = 0.0
    
    def update(self, new_value):
        """
        Updates the filter with a new value and returns the filtered value.
        
        Parameters:
            new_value (float): The new input value to filter.
        
        Returns:
            float: The filtered value.
        """
        # Apply the filter update
        filtered_value = self.b[0] * new_value + self.b[1] * self.prev_input - self.a[1] * self.prev_output
        
        # Store the current input and output for the next iteration
        self.prev_input = new_value
        self.prev_output = filtered_value
        
        return filtered_value
    
    def reset(self):
        """
        Resets the filter state (previous input and output values).
        """
        self.prev_input = 0.0
        self.prev_output = 0.0

if __name__ == '__main__':
    # Define filter parameters
    cutoff = 2.0  # Cutoff frequency in Hz
    fs = 50.0     # Sampling frequency in Hz
    order = 4     # Filter order

    # Create the low-pass filter
    lpf = LowPassFilter(cutoff, fs, order)

    # Simulate incoming values and apply the filter iteratively
    input_values = [0.5, 0.6, 0.4, 0.2, 0.8, 1.0, 0.7]  # Example input values

    filtered_values = []

    for value in input_values:
        filtered_value = lpf.update(value)
        filtered_values.append(filtered_value)

    # Print the results
    print("Input values:", input_values)
    print("Filtered values:", filtered_values)