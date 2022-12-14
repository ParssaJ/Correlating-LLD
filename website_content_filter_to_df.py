import os
import time
import bs4
import pandas as pd
import numpy as np


def create_large_set_of_english_words():
    from nltk.corpus import words
    from nltk.corpus import wordnet
    from nltk.corpus import brown

    english_brown = set(map(str.lower, brown.words()))
    english_word = set(map(str.lower, words.words()))
    english_wordnet = set(map(str.lower, wordnet.words()))
    english_words_total = english_word.union(english_wordnet, english_brown)

    """
    there are 56 words in our domainkeyword list which are not apparent in the created set.
    We will append them manually
    """

    naics_keywords = pd.read_csv("csv_files/naics_domain_list.csv")
    keywords_to_insert = []
    for col in naics_keywords.columns:
        tmp = naics_keywords[col].dropna()
        keywords_to_insert.append(tmp.values)
    keywords_to_insert = [keyword for keywords in keywords_to_insert for keyword in keywords]
    keywords_to_insert_to_set = set(keywords_to_insert)

    return english_words_total.union(keywords_to_insert_to_set)


if __name__ == '__main__':
    start = time.time()
    english_words_only = create_large_set_of_english_words()

    too_generic_keywords = pd.read_csv("csv_files/too_generic_keywords.csv", header=None).values

    training_data = pd.read_csv("csv_files/training_data.csv")
    training_data["website_content"] = ""

    for index, row in training_data.iterrows():
        print(f"Currently at index: {index}")
        webdomain = row["Webdomains"]
        if not os.path.exists("../website_responses/" + webdomain):
            training_data.at[index, "website_content"] = ""
        else:
            with open("../website_responses/" + webdomain, 'r') as file:
                webdomain_html = file.read()
                html_content_parsed = bs4.BeautifulSoup(webdomain_html, 'lxml')
                domain_keywords = [token.lower() for tokens in html_content_parsed.stripped_strings
                                   for token in tokens.split()
                                   if len(token) >= 3
                                   and token not in too_generic_keywords
                                   and token in english_words_only
                                   and token.isalpha()]
                domain_keywords = list(dict.fromkeys(domain_keywords))
                if len(domain_keywords) >= 50:
                    print(f"{len(domain_keywords)} keywords were found")
                    training_data.at[index, "website_content"] = domain_keywords

    training_data["length_of_content"] = training_data["website_content"].apply(lambda x: len(x))
    training_data = training_data[training_data["length_of_content"] >= 1]
    training_data.drop(columns="length_of_content", inplace=True)
    training_data.to_csv("csv_files/training_data_with_parsed_content.csv", index=False)
    end = time.time()
    print(f"Took a total of {np.round(((end - start) / 60), 2)} minutes")
