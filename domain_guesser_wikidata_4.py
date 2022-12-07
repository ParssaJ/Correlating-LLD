import pandas as pd
from ast import literal_eval
import numpy as np
import time


if __name__ == '__main__':
    start = time.time()
    too_generic_keywords = pd.read_csv("csv_files/too_generic_keywords.csv", header=None)
    all_domain_keywords = pd.read_csv("csv_files/naics_domain_list.csv")
    training_data = pd.read_csv("csv_files/training_data.csv")
    training_data['wikidata_keywords'] = training_data['wikidata_keywords'].apply(literal_eval)

    training_data["occurence"] = np.nan
    training_data["guessed-domain"] = ""
    training_data["keywords_with_match"] = ""

    for index, row in training_data.iterrows():
        print(f"Currently at index: {index}")
        keywords = row['wikidata_keywords']
        max_ocurrence = 0
        column = ""
        list_of_matches = []
        for col in all_domain_keywords.columns:
            occurence = 0
            current_column = pd.DataFrame(all_domain_keywords[col])
            current_column.dropna(inplace=True)
            for keyword in keywords:
                if keyword in current_column.values and keyword not in too_generic_keywords.values:
                    print(f"Match-found: {keyword}, in: {col}")
                    occurence += 1
                    list_of_matches.append(keyword)
            if occurence >= max_ocurrence:
                max_ocurrence = occurence
                column = col
        if max_ocurrence == 0:
            column = 'Other'
        print(f"Occurence: {max_ocurrence}")
        training_data.at[index, "occurence"] = max_ocurrence
        training_data.at[index, "guessed-domain"] = column
        training_data.at[index, "keywords_with_match"] = list_of_matches

    training_data.to_csv("csv_files/training_data.csv", index=False)
    end = time.time()
    print(f"Took a total of {np.round(((end - start) / 60), 2)} minutes")
