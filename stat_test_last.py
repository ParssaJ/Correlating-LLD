import pandas as pd
from PIL import Image
import numpy as np
import time
import colorsys
from scipy.stats import normaltest, kruskal
from scikit_posthocs import posthoc_nemenyi
from matplotlib import pyplot as plt


def get_hsv_values(index_string):
    with Image.open("../LLD_favicons_full_png/" + index_string + ".png") as image:
        pixels = list(zip(*image.getcolors(maxcolors=32 * 32)))
        rgb_vals = [(pixels[1][i],) * pixels[0][i] for i in range(len(pixels[0]))]
        transformed_hsv = [colorsys.rgb_to_hsv((color[0]/255), (color[1]/255), (color[2]/255))
                           for colors in rgb_vals for color in colors]
        return transformed_hsv


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

    data['hsv_values'] = data["index_in_pkl_file"].apply(get_hsv_values)

    hue_list = data['hsv_values'].tolist()
    hue_list = [hue for hues in hue_list for hue in hues]
    hue_list_test = [hue[0] for hue in hue_list]
    stat, pval = normaltest(hue_list_test)

    if pval <= .05:
        print(f"Hue Values are likely NOT normal distributed")
    else:
        print(f"Hue Values are likely normal distributed")

    # Maybe the filtered hue-Values are normal-distributed?
    hue_list_test = [hsv[0] for hsv in hue_list if hsv[1] >= .95 and hsv[2] >= .5]
    stat, pval = normaltest(hue_list_test)

    if pval <= .05:
        print(f"FILTERED Hue Values are likely NOT normal distributed")
    else:
        print(f"FILTERED Hue Values are likely normal distributed")

    groups = []
    domains = list(dict.fromkeys(data["guessed-domain"]))

    for domain in domains:
        domain_df = data[data["guessed-domain"] == domain]["hsv_values"]
        domain_hsv = domain_df.explode("hsv_values").tolist()
        filtered_hue_values = [hue for hue, sat, val in domain_hsv if sat >= .9 and val >= .5]
        groups.append(filtered_hue_values)

    # The normal-distribution of the hue-values is not given, use the kruskal-wallis-test
    stat, pval = kruskal(groups[0], groups[1], groups[2], groups[3], groups[4],
                         groups[5], groups[6], groups[7], groups[8], groups[9],
                         groups[10], groups[11], groups[12], groups[13],
                         groups[14], groups[15], groups[16], groups[17])

    if pval <= 0.05:
        print(f"The Population-Median is likely different in each group")
    else:
        print(f"The Population-Median is likely (almost) equal in each group")

    # Nemenyi test
    nemenyi = posthoc_nemenyi(groups)
    plt.matshow(nemenyi)
    plt.colorbar()
    plt.show()
    nemenyi = nemenyi < .05
    nemenyi = nemenyi[nemenyi == False]
    nemenyi = nemenyi.unstack().dropna().index.tolist()
    list_of_not_different_pairs = []
    for pair in nemenyi:
        first_el, sec_el = pair
        if pair not in list_of_not_different_pairs \
                and (sec_el, first_el) not in list_of_not_different_pairs:
            first_num, second_num = pair
            first_domain = domains[first_num-1]
            second_domain = domains[second_num-1]
            if first_domain != second_domain:
                list_of_not_different_pairs.append((first_domain, second_domain))
    print(f"List of not stat. different groups, {len(list_of_not_different_pairs)} (in total): ")
    print(list_of_not_different_pairs)
    print(domains)
    end = time.time()
    print(f"Took a total of {np.round(((end - start) / 60), 2)} minutes")
