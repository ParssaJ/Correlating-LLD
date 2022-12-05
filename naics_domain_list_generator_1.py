import numpy as np
import pandas as pd
import nltk
import os


def delete_value_from_column(df, column, old_value):
    df.loc[df[column] == old_value] = np.nan


def keep_only_nouns(token_list):
    return [domain for domain, acronym in token_list if acronym.startswith('N')]


def delete_keywords_from_column():
    return [("Real Estate & Rental & Leasing", "electronics"),
            ("Real Estate & Rental & Leasing", "video"),
            ("Real Estate & Rental & Leasing", "tape"),
            ("Real Estate & Rental & Leasing", "construction"),
            ("Real Estate & Rental & Leasing", "forestry"),
            ("Healthcare & Social Assistance", "food"),
            ("Professional, Scientific & Technical Services", "buying"),
            ("Wholesale Trade", "forestry"),
            ("Wholesale Trade", "mining"),
            ("Wholesale Trade", "agents"),
            ("Wholesale Trade", "brokers"),
            ("Retail Trade", "power"),
            ("Retail Trade", "health"),
            ("Retail Trade", "care"),
            ("Retail Trade", "clubs"),
            ("Retail Trade", "art"),
            ("Manufacturing", "power"),
            ("Manufacturing", "construction"),
            ("Manufacturing", "mining"),
            ("Manufacturing", "media"),
            ("Manufacturing", "energy"),
            ("Manufacturing", "communication")]


def domains_of_interest_with_splitting_attributes():
    return [("Agriculture, Forestry, Fishing & Hunting", "0", "21"),
            ("Mining", "21", "22"),
            ("Utilities", "22", "23"),
            ("Construction", "23", "31"),
            ("Manufacturing", "31", "41"),
            ("Wholesale Trade", "41", "441"),
            ("Retail Trade", "441", "48"),
            ("Transportation & Warehousing", "48", "51"),
            ("Information", "51", "52"),
            ("Finance & Insurance", "52", "53"),
            ("Real Estate & Rental & Leasing", "53", "54"),
            ("Professional, Scientific & Technical Services", "54", "55"),
            ("Administration, Business Support & Waste Management Services", "55", "61"),
            ("Educational Services", "61", "62"),
            ("Healthcare & Social Assistance", "62", "71"),
            ("Arts, Entertainment & Recreation", "71", "72"),
            ("Accommodation & Food Services", "72", "81"),
            ("Public Administration", "91", "919110")]


def split_domains(domain_dataframe, lower_bound, upper_bound, naics_code="code", domain_selection="name"):
    result_series = domain_dataframe.loc[
        (domain_dataframe[naics_code] >= lower_bound) &
        (domain_dataframe[naics_code] < upper_bound)] \
        [domain_selection]

    result_series = result_series[:, ].apply(str.lower)
    result_series = result_series[:, ].apply(nltk.word_tokenize)
    result_series = result_series[:, ].apply(nltk.pos_tag)
    result_series = result_series[:, ].apply(keep_only_nouns)

    result_dataframe = pd.DataFrame(result_series)
    result_dataframe = result_dataframe.explode(domain_selection)
    result_dataframe.drop_duplicates(inplace=True)
    result_dataframe.reset_index(drop=True, inplace=True)
    return result_dataframe


if __name__ == '__main__':

    if os.path.exists("csv_files/naics.csv"):
        df = pd.read_csv("csv_files/naics.csv")
    else:
        raise FileNotFoundError("Please create a csv_files directory "
                                "and move the naics.csv file to that location")

    df = df.astype(str)

    top_level_domains = domains_of_interest_with_splitting_attributes()
    list_of_domains = []
    for domain, lower_bound, upper_bound in top_level_domains:
        df[domain] = split_domains(df, lower_bound, upper_bound)
        list_of_domains.append(domain)

    only_domains_df = df[list_of_domains]

    keywords_to_delete_from_column = delete_keywords_from_column()
    for keyword in keywords_to_delete_from_column:
        delete_value_from_column(only_domains_df, keyword[0], keyword[1])

    # Sort the columns by the number of elements
    only_domains_df = only_domains_df[only_domains_df.isna().sum().sort_values().keys()]
    only_domains_df.replace(regex=True, value='theater', to_replace=r".*theatre", inplace=True)
    only_domains_df.replace(regex=True, value='theaters', to_replace=r".*theatres", inplace=True)
    only_domains_df_final = only_domains_df.dropna(axis=0, how='all')
    only_domains_df_final.to_csv("csv_files/naics_domain_list.csv", index=False)
