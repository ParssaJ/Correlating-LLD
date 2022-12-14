import pandas as pd
from PIL import Image
import numpy as np
import time
from scipy.stats import normaltest, kruskal
from scikit_posthocs import posthoc_dunn
from matplotlib import pyplot as plt


def get_rgb_values(index_string):
    with Image.open("../LLD_favicons_full_png/" + index_string + ".png") as image:
        pixels = list(zip(*image.getcolors(maxcolors=32*32)))
        rgb_vals = [(pixels[1][i],)*pixels[0][i] for i in range(len(pixels[0]))]
        transformed_rgb = [np.log2(int("".join(map(str, color)))+1)
                           for colors in rgb_vals
                           for color in colors]
        return transformed_rgb


if __name__ == '__main__':
    start = time.time()

    training_data = pd.read_csv("csv_files/training_data.csv")
    data = training_data[["Webdomains", "guessed-domain"]]
    data = data.assign(index_in_pkl_file="")
    data = data.assign(rgb_values="")

    meta_data = pd.read_pickle("LLD-icon-full_data-names.pkl")
    meta_data = meta_data.astype(str)
    meta_data = pd.DataFrame(meta_data.reshape(-1, 1)).rename(columns={0: "domain_name"})
    data['index_in_pkl_file'] = data["Webdomains"].apply(lambda x:
                                                         f"{meta_data[meta_data['domain_name'] == x].index.values[0]:06}")

    data['rgb_values'] = data["index_in_pkl_file"].apply(get_rgb_values)

    groups = []
    domains = list(dict.fromkeys(data["guessed-domain"]))
    for domain in domains:
        domain_df = data[data["guessed-domain"] == domain]["rgb_values"]
        domain_rgb = domain_df.explode("rgb_values").tolist()
        groups.append(domain_rgb)

        stat, pval = normaltest(domain_rgb)

        if pval <= 0.05:
            print(f"Domain: {domain} is likely NOT normal distributed")
        else:
            print(f"Domain: {domain} is likely normal distributed")

    # According to the normaltest none of the groups are normal-distributed
    # Use the Kruskal-Wallis-Test("nonparametric-equivalent of ANOVA") instead
    stat, pval = kruskal(groups[0], groups[1], groups[2], groups[3], groups[4],
                         groups[5], groups[6], groups[7], groups[8], groups[9],
                         groups[10], groups[11], groups[12], groups[13],
                         groups[14], groups[15], groups[16], groups[17])

    if pval <= 0.05:
        print(f"The Median is likely different in each group")
    else:
        print(f"The Median is likely (almost) equal in each group")

    # Dunnet-Test
    dunn = posthoc_dunn(groups, p_adjust='bonferroni')
    plt.matshow(dunn)
    plt.colorbar()
    plt.show()

    end = time.time()
    print(f"Took a total of {np.round(((end - start) / 60), 2)} minutes")
