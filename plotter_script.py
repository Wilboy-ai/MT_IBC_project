import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import make_interp_spline
import natsort
import re

def read_csv_data(path_to_file):
    with open(path_to_file, newline='') as csvfile:
        all_data = []
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            _row = []
            for elem in row:
                _row.append(float(elem))
            all_data.append(_row)
        all_data = np.array(all_data)

        #print(all_data.T)
    return all_data.T

def split_data_into_runs(data, column):
    splitted_runs = []
    one_run = []

    for i in range(len(data[column,:])):
        one_run.append(data[column, :][i])

        if len(one_run) >= 25:
            splitted_runs.append(one_run)
            one_run = []

    #print(splitted_runs)
    return splitted_runs

def plot_csv_metric(path_to_file, title="Unknown"):
    data = read_csv_data(path_to_file)
    data = split_data_into_runs(data, 1)

    success_runs = 0
    total_runs = 0

    num_runs = len(data)
    legends_titles = []
    for i in range(0, num_runs):
        #print(f'Max: {np.max(data[i])}, median {np.median(data[i])}, last data {data[i][-10:]}')

        #if np.mean(data[i][-3:]) >= 0.9:
        if np.median(data[i][-10:]) >= 0.95:
            plt.plot(data[i], linewidth=1, alpha=1.)
            #legends_titles.append(f'run {i + 1} (success), {np.mean(data[i][-3:])}')
            success_runs += 1
        else:
            #plt.plot(data[i],'--', linewidth=0.35, alpha=0.5)
            legends_titles.append(f'run {i + 1} (failed), {np.mean(data[i][-3:])}')

        total_runs += 1

    for runs in legends_titles:
        print(runs)

    print(f'success rate: {(success_runs/total_runs)*100}%')

    #plt.title(title)
    #plt.ylabel('return')
    #plt.xlabel('steps')
    #plt.legend(legends_titles, fontsize=5)
    #plt.show()


def plot_demos(title="Unknown", y_label="unknwon", column=1):
    # setting font sizeto 30
    plt.subplots(figsize=(20, 10))

    csv_files = glob.glob('csv_files/suture_throw_demo_*.csv')
    csv_files = natsort.natsorted(csv_files)
    all_data = []
    legends_titles = []

    def running_average(data, window):
        smooth_data = []
        for ind in range(len(data) - window + 1):
            smooth_data.append(np.mean(data[ind:ind + window]))
        return smooth_data

    for file_name in csv_files:
        print(file_name)
        data = read_csv_data(file_name)
        data = split_data_into_runs(data, column)
        all_data.append(data[0])
        print(data[0])


    run_index = 20
    for _data in all_data:
        #X_Y_Spline = make_interp_spline(np.linspace(0, 25, 25), _data)
        #X_ = np.linspace(0, 25, 500)
        #Y_ = X_Y_Spline(X_)
        smooth_data = running_average(data=_data, window=5)
        if np.mean(_data[-10:]) >= 0.95:
            plt.plot(smooth_data, linewidth=2, alpha=1)
            legends_titles.append(f'run {run_index} (success)')
        else:
            plt.plot(smooth_data, '--', linewidth=0.5, alpha=0.3)
            legends_titles.append(f'run {run_index} (failed)')
        run_index += 1

    plt.title(title, fontsize=30)
    #plt.legend(legends_titles, fontsize=10)
    plt.ylabel(y_label, fontsize=25)
    plt.xlabel('steps', fontsize=25)
    plt.savefig("figures/demos_noisy_states_std_001.png", bbox_inches="tight")
    plt.show()

def collect_all_data(paths, output_file):
    plt.subplots(figsize=(20, 10))

    fail_count = 0
    success_count = 0
    def running_average(data, window):
        smooth_data = []
        for ind in range(len(data) - window + 1):
            smooth_data.append(np.mean(data[ind:ind + window]))
        return smooth_data

    output_csv_file = "noisy_states_data_001_collected.csv"
    legends_titles = []
    for path in paths:
        csv_files = glob.glob(f'{path}/Suture_eval_*.csv')
        csv_files = natsort.natsorted(csv_files)
        for file in csv_files:
            print(file)
            run_name = re.findall(r'\d+', file)

            with open(file, newline='') as csvfile:
                all_data = []
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    _row = []
                    for elem in row:
                        _row.append(float(elem))
                    all_data.append(_row)
                all_data = np.array(all_data).T

            all_runs = all_data[1]
            one_run = []
            run_index = 1

            for elem in all_runs:
                one_run.append(elem)
                if len(one_run) >= 25:
                    smooth_data = running_average(data=one_run, window=5)
                    if np.mean(one_run[-10:]) >= 0.95:
                        plt.plot(smooth_data, linewidth=2, alpha=1.0, label=f'run:{run_name[0]} itr:{run_name[1]}')
                        #legends_titles.append(f'{file}-{run_index} (success)')

                        success_count += 1
                    else:
                        plt.plot(smooth_data, '--', linewidth=0.8, alpha=0.5, label=None)
                        #legends_titles.append(f'{run_name[-1]}-{run_index} (failed)')
                        fail_count += 1

                    run_index += 1
                    one_run = []

    print(f"Estimated Successrate: {(success_count/(success_count+fail_count))*100} %")
    plt.title("Noisy states std=.001", fontsize=30)
    #plt.legend(legends_titles, fontsize=5)
    plt.legend()
    plt.ylabel('return', fontsize=25)
    plt.xlabel('steps', fontsize=25)

    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


def plot_eval_episode(dir):
    plt.subplots(figsize=(10, 10))

    csv_files = glob.glob(f'{dir}/Suture_eval_*.csv')
    csv_files.sort()
    all_data = []
    legends_titles = []

    for file_name in csv_files:
        print(file_name)
        data = read_csv_data(file_name)
        data = split_data_into_runs(data, column)
        all_data.append(data[0])
        print(data[0])


def count_successrate(paths):
    results = []

    fail_count = 0
    success_count = 0

    for path in paths:
        csv_files = glob.glob(f'{path}/Suture_eval_*.csv')
        csv_files = natsort.natsorted(csv_files)
        for file in csv_files:
            with open(file, newline='') as csvfile:
                all_data = []
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    _row = []
                    for elem in row:
                        _row.append(float(elem))
                    all_data.append(_row)
                all_data = np.array(all_data).T

            all_runs = all_data[1]
            one_run = []
            run_index = 1
            run_means = []
            for elem in all_runs:
                one_run.append(elem)
                if len(one_run) >= 25:
                    if np.mean(one_run[-5:]) >= 0.9:
                        success_count += 1
                    else:
                        fail_count += 1
                    run_index += 1
                    run_means.append(np.mean(one_run[-5:]))
                    one_run = []
            results.append((file, (success_count / (success_count + fail_count)) * 100, run_means))
            success_count = 0
            fail_count = 0

    return results


    #return successrate, videos
    #return 0



#path = "csv_files/suture_throw_demo_20.csv"
#plot_title = "unknown"
#plot_csv_metric(path_to_file=path, title=plot_title)


#path = "noise_states-1/Suture_eval_50000.csv"
#plot_title = "Suture_eval_50000"
#plot_csv_metric(path_to_file=path, title=plot_title)

#plot_demos(title="Demonstrations noisy states (std=0.001)", y_label="return",column=1)

#plot_eval_episode("noise_states-1")

#collect_all_data(["noise_states-1", "noise_states-2","noise_states-3", "noise_states-4", "noise_states-5"])

#collect_all_data(["noise_states-5", "noise_states-2"], "figures/noisy_states_std_001_20_demos.png")
#collect_all_data(["noise_states-5", "noise_states-6", "noise_states-7", "noise_states-8", "noise_states-9", "noise_states-10"], "figures/noisy_states_std_001_50_demos.png")

def plot_noise_data():
    def running_average(data, window):
        smooth_data = []
        for ind in range(len(data) - window + 1):
            smooth_data.append(np.mean(data[ind:ind + window]))
        return smooth_data

    plt.subplots(figsize=(5, 5))
    X = np.linspace(50, 50000, 11)
    plt.title("Noisy states performance")

    demos_20 = []
    demos_50 = []
    for i in range(1,6):
        results = count_successrate([f"noise_states-{i}"])
        file, score, acc = zip(*results)
        demos_20.append(list(score))
        #print(np.array(acc))
        #plt.plot(X, np.array(acc), '--', linewidth=1.8, alpha=0.9, label=None)
        #break

        plt.plot(X, np.mean(np.array(acc),axis=1), '--', linewidth=1.0, alpha=0.5, label=None)

    for i in range(6,11):
        results = count_successrate([f"noise_states-{i}"])
        file, score, acc = zip(*results)
        demos_50.append(list(score))

        #print(np.array(acc))
        #plt.plot(X, np.array(acc), '--', linewidth=1., alpha=0.9, label=None)
        plt.plot(X, np.mean(np.array(acc), axis=1), '-', linewidth=2.0, alpha=1.0, label=None)


    demos_20 = np.array(demos_20)
    demos_50 = np.array(demos_50)

    mean_data_20 = np.mean(demos_20, axis=0)
    mean_data_50 = np.mean(demos_50, axis=0)

    #plt.plot(X, demos_20.T, '--', linewidth=0.8, alpha=0.3, color=[1,0.25,0], label=None)
    #plt.plot(X, demos_50.T, '--', linewidth=0.8, alpha=0.3, color=[0,0,1], label=None)
    #plt.plot(X, mean_data_20, linewidth=2., color=[1,0.25,0], label='20 demonstrations')
    #plt.plot(X, mean_data_50, linewidth=2., color=[0,0,1], label='50 demonstrations')
    plt.xlabel('Iterations')
    plt.ylabel('Success rate')
    plt.legend()
    plt.grid()
    plt.show()




path = "noisy_states_eval/Suture_eval.csv"
plot_title = "unknown"
plot_csv_metric(path_to_file=path, title=plot_title)
