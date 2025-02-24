import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# endregion

# region functions
def truncated_lognormal_pdf(x, mu, sigma, D_min, D_max):
    """ Calculate the PDF of the truncated log-normal distribution. """
    coeff = 1 / (stats.lognorm(sigma, scale=np.exp(mu)).sf(D_min) - stats.lognorm(sigma, scale=np.exp(mu)).sf(D_max))
    pdf = coeff * stats.lognorm(sigma, scale=np.exp(mu)).pdf(x)
    return pdf


def truncated_lognormal_cdf(x, mu, sigma, D_min, D_max):
    """ Calculate the CDF of the truncated log-normal distribution. """
    coeff = 1 / (stats.lognorm(sigma, scale=np.exp(mu)).sf(D_min) - stats.lognorm(sigma, scale=np.exp(mu)).sf(D_max))
    cdf = coeff * (stats.lognorm(sigma, scale=np.exp(mu)).cdf(x) - stats.lognorm(sigma, scale=np.exp(mu)).cdf(D_min))
    return cdf


def main():
    # Get user input for parameters
    mu = float(input("Enter the mean (mu) for the log-normal distribution: "))
    sigma = float(input("Enter the standard deviation (sigma) for the log-normal distribution: "))
    D_min = float(input("Enter the minimum D for the truncated distribution: "))
    D_max = float(input("Enter the maximum D for the truncated distribution: "))

    # Calculate the value at D_min + (D_max - D_min) * 0.75
    D_target = D_min + (D_max - D_min) * 0.75

    # Generate x values for the plot
    x = np.linspace(D_min, D_max, 500)

    # Calculate PDF and CDF
    pdf_values = truncated_lognormal_pdf(x, mu, sigma, D_min, D_max)
    cdf_values = truncated_lognormal_cdf(x, mu, sigma, D_min, D_max)

    # Plot PDF
    plt.figure(figsize=(10, 8))

    # Plotting the PDF
    plt.subplot(2, 1, 1)
    plt.plot(x, pdf_values, label='Truncated Log-Normal PDF', color='blue')
    plt.fill_between(x, pdf_values, where=(x >= D_min) & (x <= D_target), color='grey', alpha=0.3)

    # Annotations and labels for PDF
    plt.title('Truncated Log-Normal Probability Density Function')
    plt.xlabel('D')
    plt.ylabel('f(D)')
    plt.axvline(D_target, color='red', linestyle='--')
    plt.annotate(f'F(D) at D={D_target:.2f}', xy=(D_target, max(pdf_values) * 0.5),
                 xytext=(D_target + 1, max(pdf_values) * 0.6),
                 arrowprops=dict(arrowstyle='->', color='red'))
    plt.xlim(D_min, D_max)
    plt.ylim(0, max(pdf_values) * 1.1)
    plt.grid()
    plt.legend()

    # Plotting the CDF
    plt.subplot(2, 1, 2)
    plt.plot(x, cdf_values, label='Truncated Log-Normal CDF', color='green')
    plt.axvline(D_target, color='red', linestyle='--')
    plt.title('Truncated Log-Normal Cumulative Distribution Function')
    plt.xlabel('D')
    plt.ylabel('F(D)')
    plt.xlim(D_min, D_max)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# endregion

# region function calls
if __name__ == "__main__":
    main()
