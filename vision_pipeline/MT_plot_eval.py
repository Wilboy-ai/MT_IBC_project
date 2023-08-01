import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import kstest
from scipy import stats
from scipy.spatial.transform import Rotation


figure_size = (4, 4)
bin_size = 45

def save_error_distributions(csv_file):

    ktest_table_data = [["Dimension", "result.statistic", "critical_value", "normal_test", "alpha"]]
    dims_metrics = [["Dimension", "Mean", "Median", "Std", "min", "max"]]

    dims_names = ["x", "y", "z", "x_orient", "y_orient", "z_orient", "w_orient"]
    file_name = csv_file
    file_name = file_name.replace(".csv", "")
    file_name = file_name.replace("evaluation_csv/", "")

    for i in range(0, 7):
        fig, ax = plt.subplots(figsize=figure_size)
        # Estimate the parameters of the normal distribution
        mean, std = norm.fit(all_error[:, i])
        # Plot the estimated normal distribution
        x = np.linspace(all_error[:, i].min(), all_error[:, i].max(), 100)
        #x = np.linspace(-0.5, 0.5, 100)
        y = norm.pdf(x, mean, std)
        ax.plot(x, y, 'b-', linewidth=2)

        #weights = np.ones_like(all_error[:, i]) / len(all_error[:, i])
        ax.hist(all_error[:, i],density=True, bins=bin_size, alpha=0.3)

        #result = kstest(all_error[:, i], norm.cdf)
        # Perform the Kolmogorov-Smirnov test
        result = kstest(all_error[:, i], 'norm')
        dims_metrics.append([dims_names[i], mean, np.median(all_error[:, i]), std, all_error[:, i].min(), all_error[:, i].max()])

        # Calculate the critical value at the chosen significance level
        alpha = 0.01
        critical_value = norm.ppf(1 - alpha / 2) / np.sqrt(len(all_error[:, i]))
        #print(f'result.statistic: {result.statistic}, critical_value:{critical_value}')
        # Print the test result
        if result.statistic < critical_value:
            #print("Data follows a normal distribution (fail to reject H0)")
            normal_test = "fail to reject H0: normal distribution"
        else:
            #print("Data does not follow a normal distribution (reject H0)")
            normal_test = "reject H0: normal distribution"
        ktest_table_data.append([dims_names[i], result.statistic, critical_value, normal_test, alpha])


        # Add mean and standard deviation to the subplot
        ax.text(0.02, 0.98, f"Mean: {mean:.5f}\nStd: {std:.5f}", transform=ax.transAxes, fontsize=10,
                ha='left', va='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
        # Add subtitle to the subplot
        ax.set_title(f'{dims_names[i]}')
        plt.xlabel('Error', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        fig.tight_layout()
        fig.savefig(f'evaluation_figures/{file_name}-{dims_names[i]}.jpg')
        plt.close()


        plot_title = f'{file_name}-{dims_names[i]}'
        plot_title = plot_title.replace("_", " ")
        plot_title = plot_title.replace("-", " ")

        # Create QQ plot
        fig, ax = plt.subplots()
        stats.probplot(all_error[:, i], dist='norm', plot=ax)
        # Set labels and title
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.title(f'QQ Plot ({plot_title})')
        # Show the plot
        #plt.show()
        fig.tight_layout()
        fig.savefig(f'evaluation_figures/{file_name}-{dims_names[i]}-QQ.jpg')
        plt.close()


    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(14,3))
    # Create the table
    table = ax.table(cellText=ktest_table_data, loc='center', cellLoc='center')
    # Modify the table properties
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust the table size
    plotname = file_name.replace('_eval', '')
    plotname = plotname.replace('_', ' ')
    ax.set_title(f'ktest results for {plotname}')
    # Hide the axes
    ax.axis('off')
    # Show the plot
    #plt.show()
    fig.tight_layout()
    fig.savefig(f'evaluation_figures/{file_name}-ktest.jpg')
    plt.close()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(14, 3))
    # Create the table
    table = ax.table(cellText=dims_metrics, loc='center', cellLoc='center')
    # Modify the table properties
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust the table size
    plotname = file_name.replace('_eval', '')
    plotname = plotname.replace('_', ' ')
    ax.set_title(f'Dimensions metrics ({plotname})')
    # Hide the axes
    ax.axis('off')
    # Show the plot
    # plt.show()
    fig.tight_layout()
    fig.savefig(f'evaluation_figures/{file_name}-dims_metrics.jpg')
    plt.close()





def save_error_distributions_MAE(csv_file):
    file_name = csv_file
    file_name = file_name.replace(".csv", "")
    file_name = file_name.replace("evaluation_csv/", "")

    plot_names = ["Translational_error", "Orientational_error"]
    plot_title = ["Translational error", "Orientational error"]
    i = 0

    # for label, pred in zip(true_labels, pred_labels):
    #     print(f'{label[:3]} - {pred[:3]}')
    #
    #     abs_error = np.array(label[:3]) - np.array(pred[:3])
    #     print(abs_error)

    output_string = ""

    MAE_trans = all_error[:, :3]
    MAE_orient = all_error[:, 3:7]
    for MAE in [MAE_trans, MAE_orient]:
        if i > 0:
            absolute_erros = []
            for quat in MAE:
                rotation = Rotation.from_quat(quat)
                #axis_angle = rotation.as_rotvec()
                axis_angle = rotation.as_euler('xyz')
                angle = np.linalg.norm(axis_angle)
                absolute_erros.append(angle)
            #print(f'orient: {absolute_erros}')
        else:
            absolute_erros = np.linalg.norm(MAE, axis=1)
            absolute_erros= absolute_erros.tolist()
            #print(f'trans: {absolute_erros}')

        fig, ax = plt.subplots(figsize=figure_size)

        # Estimate the parameters of the normal distribution
        mean, std = norm.fit(absolute_erros)
        # Plot the estimated normal distribution
        #x = np.linspace(absolute_erros.min(), absolute_erros.max(), 100)
        #y = norm.pdf(x, mean, std)
        #ax.plot(x, y, 'b-', linewidth=2)
        abs_median_error = np.median(absolute_erros)

        print(f'{plot_title[i]}: & {mean:.5f} ({abs_median_error:.5f})')


        # Add vertical lines for mean and median
        ax.axvline(x=mean, color='red', linestyle='--', label='Mean')
        ax.axvline(x=abs_median_error, color='blue', linestyle='-.', label='Median')

        #weights = np.ones_like(absolute_erros) / len(absolute_erros)
        counts, bins, _ = ax.hist(absolute_erros, density=True, bins=bin_size, alpha=0.9)

        # Add mean and standard deviation to the subplot
        ax.text(0.03, 0.97, f"Mean: {mean:.5f}\nMedian: {abs_median_error:.5f}\nStd: {std:.5f}", transform=ax.transAxes, fontsize=10,
                ha='left', va='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
        # Add subtitle to the subplot
        model_name = file_name.replace("_eval", "")
        model_name = model_name.replace("_", " ")
        ax.set_title(f'{model_name} {plot_title[i]}')
        plt.xlabel('Error', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)

        # Add a legend
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(f'evaluation_figures/{file_name}_{plot_names[i]}.jpg')
        i += 1
        plt.close()



csv_files = []
for file in os.listdir("evaluation_csv"):
    csv_files.append("evaluation_csv" + "/" + file)
    # print(file)
print(f'Total files: {len(csv_files)}')

for csv_file in csv_files:
    true_labels = []
    pred_labels = []
    all_error = []


    with open(csv_file, 'r', newline='') as csvfile:
        print(csv_file)
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in spamreader:
            true_label = np.array(row[:7])
            pred_label = np.array(row[7:])
            error = true_label - pred_label
            true_labels.append(true_label.tolist())
            pred_labels.append(pred_label.tolist())
            all_error.append(error.tolist())

    all_error = np.array(all_error)

    save_error_distributions(csv_file)
    save_error_distributions_MAE(csv_file)




# print("Quick Sanity Test")
# # Generate random data from a normal distribution
# sample_size = 1000
# mu = 0
# sigma = 1
# data = np.random.normal(mu, sigma, sample_size)
#
# # Perform the Kolmogorov-Smirnov test
# result = kstest(data, 'norm')
#
# # Calculate the critical value at the chosen significance level
# alpha = 0.05
# critical_value = norm.ppf(1 - alpha/2) / np.sqrt(sample_size)
#
# # Print the test result
# if result.statistic < critical_value:
#     print("Data follows a normal distribution (fail to reject H0)")
# else:
#     print("Data does not follow a normal distribution (reject H0)")
#
# # Plot the data and the theoretical normal distribution
# plt.hist(data, bins=30, density=True, alpha=0.5, label='Data Histogram')
# x = np.linspace(-4, 4, 100)
# plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label='Normal Distribution')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.title('Data Distribution vs. Normal Distribution')
# plt.legend()
# plt.show()







# # Create sample data for the table
# data = [
#     ['Name', 'Age', 'Country'],
#     ['John', 25, 'USA'],
#     ['Alice', 30, 'Canada'],
#     ['Bob', 35, 'Australia']
# ]
#
# # Create the figure and axes
# fig, ax = plt.subplots()
#
# # Create the table
# table = ax.table(cellText=data, loc='center', cellLoc='center')
#
# # Modify the table properties
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1, 1.5)  # Adjust the table size
#
# # Hide the axes
# ax.axis('off')
#
# # Show the plot
# plt.show()






