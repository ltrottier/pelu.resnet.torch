import pdb
import numpy as np
import re
import os

prependpaths = [
    os.path.join("results", "results-pelu-div-mul"),
    os.path.join("results", "results-bn-pelu-div-mul"),
    os.path.join("results", "results-elu"),
    os.path.join("results", "results-bn-elu"),
    os.path.join("results", "results-bn-relu"),
    os.path.join("results", "results-bn-prelu"),
    os.path.join("results", "results-pelu-div-div"),
    os.path.join("results", "results-pelu-mul-div"),
    os.path.join("results", "results-pelu-mul-mul"),
]

depths = [110]
datasets = [
    "cifar10",
    "cifar100"
]

for prependpath in prependpaths:
    for dataset in datasets:
        medians_min = [None for _ in range(len(depths))]
        medians_mean = [None for _ in range(len(depths))]
        for i, depth in enumerate(depths):
            errors = []
            for t in range(1,6):
                resultspath = os.path.join("try{:d}", dataset, "{:d}").format(t, depth)
                filepath = os.path.join(prependpath, resultspath, 'nohup.out')
                # print(filepath)
                try:
                    with open(filepath, "r") as fh:
                        content = fh.read(-1)
                        cur_errors = re.findall("top1:\s*\d+\.\d+", content)
                        del cur_errors[-1]
                        cur_errors = [[float(re.findall("\d+\.\d+", i)[0])] for i in cur_errors]
                        cur_errors = np.array(cur_errors)

                        if len(cur_errors) != 200:
                            raise Exception("")
                        errors.append(cur_errors)
                except:
                    print("Error reading {:s}".format(filepath))

            if len(errors) != 0:
                errors_median = np.median(np.hstack(errors), 1)
                medians_mean[i] = errors_median[-5:].mean()
                medians_min[i] = errors_median.min()

        print("{:s} - {}: mean {}, min {}".format(prependpath, dataset, medians_mean, medians_min))
