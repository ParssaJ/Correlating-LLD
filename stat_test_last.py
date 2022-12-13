import pandas as pd
from scipy.stats import normaltest, kruskal
from scikit_posthocs import posthoc_dunn
from matplotlib import pyplot as plt


if __name__ == '__main__':
    rgb_vals = pd.read_csv("csv_files/training_data.csv")
    domains = list(dict.fromkeys(rgb_vals["guessed-domain"]))

    groups = []
    dunn_df = pd.DataFrame(columns=["guessed-domain", "RGB-Values"])

    for domain in domains:
        curr_domain = rgb_vals[rgb_vals["guessed-domain"] == domain]
        curr_domain_rgb = curr_domain.values[0][1].split(",")
        curr_domain_rgb = [s.replace("]", "") for s in curr_domain_rgb]
        curr_domain_rgb = [s.replace("[", "") for s in curr_domain_rgb]
        curr_domain_rgb = [float(x) for x in curr_domain_rgb]

        groups.append(curr_domain_rgb)

        tmp_df = pd.DataFrame([[domain, curr_domain_rgb]], columns=["guessed-domain", "RGB-Values"])
        dunn_df = pd.concat([dunn_df, tmp_df], ignore_index=True)

        stat, pval = normaltest(curr_domain_rgb)
        if pval <= 0.05:
            print(f"Domain: {domain} is likely normal-distributed")
        else:
            print(f"Domain: {domain} is likely NOT normal-distributed")

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
    dunn_df = dunn_df.explode("RGB-Values")
    dunn = posthoc_dunn(dunn_df, val_col='RGB-Values', group_col='Domain', p_adjust='bonferroni')
    plt.matshow(dunn)
    plt.colorbar()
    plt.show()
