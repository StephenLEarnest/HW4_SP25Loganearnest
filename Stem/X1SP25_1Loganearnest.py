import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import math


def ln_PDF(D, mu, sigma):
    """
    Computes the log-normal probability density function (PDF) for a given diameter.
    :param D: rock diameter (must be > 0)
    :param mu: mean of ln(D)
    :param sigma: standard deviation of ln(D)
    :return: value of the PDF at diameter D
    """
    # Check if the diameter is valid (greater than 0)
    if D <= 0.0:
        return 0.0

    # Calculate the PDF using the log-normal formula
    p = 1 / (D * sigma * np.sqrt(2 * np.pi))
    _exp = -((np.log(D) - mu) ** 2) / (2 * sigma ** 2)

    return p * np.exp(_exp)


def tln_PDF(D, mu, sigma, F_DMin, F_DMax):
    """
    Computes the value of the truncated log-normal PDF.
    :param D: rock diameter
    :param mu: mean of ln(D)
    :param sigma: standard deviation of ln(D)
    :param F_DMin: cumulative distribution function value at D_Min
    :param F_DMax: cumulative distribution function value at D_Max
    :return: value of the truncated PDF at diameter D
    """
    # Call the log-normal PDF and normalize it
    return ln_PDF(D, mu, sigma) / (F_DMax - F_DMin)


def F_tlnpdf(mu, sigma, D_Min, D, D_Max, F_DMax, F_DMin):
    """
    Computes the cumulative probability of the truncated log-normal distribution
    by integrating the PDF from D_Min to D.
    :param mu: mean of ln(D)
    :param sigma: standard deviation of ln(D)
    :param D_Min: minimum diameter
    :param D: diameter
    :param D_Max: maximum diameter
    :param F_DMax: cumulative distribution function value at D_Max
    :param F_DMin: cumulative distribution function value at D_Min
    :return: cumulative probability from D_Min to D
    """
    # Check if D is within the valid range
    if D > D_Max or D < D_Min:
        return 0

    # Integrate the truncated PDF from D_Min to D
    integral_result, _ = quad(tln_PDF, D_Min, D, args=(mu, sigma, F_DMin, F_DMax))

    return integral_result


def makeSample(mu, sigma, D_Min, D_Max, F_DMax, F_DMin, N=100):
    """
    Generates a sample of rock sizes using the truncated log-normal distribution.
    :param mu: mean of ln(D)
    :param sigma: standard deviation of ln(D)
    :param D_Min: minimum diameter
    :param D_Max: maximum diameter
    :param F_DMax: cumulative distribution function value at D_Max
    :param F_DMin: cumulative distribution function value at D_Min
    :param N: number of samples to generate
    :return: list of generated rock sizes
    """
    # Generate uniformly distributed probabilities
    probs = np.random.rand(N)
    d_s = []  # List to hold generated sample sizes

    for p in probs:
        # Initial guess for diameter D
        D_initial_guess = (D_Max + D_Min) / 2

        # Use fsolve to find the diameter D that gives the cumulative probability p
        D_solution = fsolve(lambda D: F_tlnpdf(mu, sigma, D_Min, D, D_Max, F_DMax, F_DMin) - p, D_initial_guess)

        # Append the found diameter to the list of samples
        d_s.append(D_solution[0])

    return d_s


def sampleStats(D):
    """
    Computes the mean and variance of a sample of values.
    :param D: a list of values (rock sizes)
    :return: (mean, variance) of the sample
    """
    mean = np.mean(D)  # Calculate mean of the sample
    variance = np.var(D, ddof=1)  # Calculate sample variance (ddof=1 for unbiased estimate)

    return mean, variance


def getFDMaxFDMin(mu, sigma, D_Min, D_Max):
    """
    Computes the cumulative distribution function values at D_Min and D_Max.
    :param mu: mean of ln(D)
    :param sigma: standard deviation of ln(D)
    :param D_Min: minimum diameter
    :param D_Max: maximum diameter
    :return: (F_DMin, F_DMax) cumulative distribution function values
    """
    # Compute cumulative probabilities using numerical integration
    F_DMax, _ = quad(ln_PDF, 0, D_Max, args=(mu, sigma))
    F_DMin, _ = quad(ln_PDF, 0, D_Min, args=(mu, sigma))

    return F_DMin, F_DMax


def main():
    """
    Main function to simulate the gravel production process and generate rock size samples.
    """
    # Set up default parameter values
    mean_ln = math.log(2)  # Mean of ln(D) in inches
    sig_ln = 1  # Standard deviation of ln(D)
    D_Max = 1.0  # Maximum diameter
    D_Min = 3.0 / 8.0  # Minimum diameter
    N_samples = 11  # Number of samples to generate
    N_sampleSize = 100  # Sample size for each sample
    goAgain = True  # Flag to control the main loop

    while goAgain:
        # Get user input for parameters, using defaults if no input is given
        mean_ln = float(input(f'Mean of ln(D) (default: {mean_ln:.3f}): ') or mean_ln)
        sig_ln = float(input(f'Standard deviation of ln(D) (default: {sig_ln:.3f}): ') or sig_ln)
        D_Min = float(input(f'Small aperture size (default: {D_Min:.3f}): ') or D_Min)
        D_Max = float(input(f'Large aperture size (default: {D_Max:.3f}): ') or D_Max)

        # Compute cumulative distribution function values for D_Min and D_Max
        F_DMin, F_DMax = getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max)

        # Generate samples and compute their statistics
        means = []  # List to store sample means
        for _ in range(N_samples):
            # Generate a sample and calculate its statistics
            sample = makeSample(mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N=N_sampleSize)
            sample_mean, sample_variance = sampleStats(sample)
            means.append(sample_mean)  # Store the mean of the sample
            print(f"Sample mean: {sample_mean:.3f}, Sample variance: {sample_variance:.3f}")

        # Compute statistics of the sample means
        overall_mean, overall_variance = sampleStats(means)
        print(f"Mean of the sampling mean: {overall_mean:.3f}")
        print(f"Variance of the sampling mean: {overall_variance:.6f}")

        # Ask user if they want to repeat the simulation
        goAgain = input('Go again? (y/n): ').strip().lower() == 'y'


if __name__ == '__main__':
    main()
