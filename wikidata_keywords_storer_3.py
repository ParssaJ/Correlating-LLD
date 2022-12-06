import sys
from datetime import datetime
import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import inflect
import nltk
import time
from urllib.error import HTTPError


def return_pluralized_nouns(nouns):
    if type(nouns) == list:
        return [inflect_engine.plural(noun) for noun in nouns]
    return inflect_engine.plural(nouns)


def remove_nan_and_empty_values_from_list(list_to_filter):
    return [keyword for keyword in list_to_filter if type(keyword) == str and keyword != '' and len(keyword) > 2
            and keyword != 'nan' and keyword != ' ' and keyword != 'http' and keyword != 'https' and keyword.isalpha()]


def create_list_from_dataframe_and_drop_duplicates(dataframe):
    tmp_list = []
    for col in dataframe.columns:
        tmp_list += tmp_list + dataframe[col].tolist()
    return list(dict.fromkeys(tmp_list))


def keep_only_nouns(token_list):
    return [domain for domain, acronym in token_list if acronym.startswith('N')]


def tokenize_and_keep_only_nouns(df_to_transform):
    result = pd.DataFrame()
    df_to_transform = df_to_transform.applymap(str)
    df_to_transform = df_to_transform.applymap(str.lower)
    df_to_transform = df_to_transform.applymap(nltk.word_tokenize)
    df_to_transform = df_to_transform.applymap(nltk.pos_tag)
    df_to_transform = df_to_transform.applymap(keep_only_nouns)
    for col in df_to_transform.columns:
        tmp_column = df_to_transform[col]
        tmp_column_df = pd.DataFrame(tmp_column)
        tmp_column_df_exploded = tmp_column_df.explode(col)
        tmp_column_df_exploded.drop_duplicates(inplace=True)
        tmp_column_df_exploded.reset_index(inplace=True, drop=True)
        result = pd.concat([result, tmp_column_df_exploded], axis=1)
    return result


def search_in_wikidata(webdomains):
    unpacked_webdomain_string = ""
    for i in range(len(webdomains)):
        unpacked_webdomain_string += webdomains[i][0]
        unpacked_webdomain_string += webdomains[i][1]

    query_string = "SELECT DISTINCT ?itemLabel ?keywordLabel ?keywordAltLabel  ?itemdescription ?found_domain \
                    WHERE { \
                        VALUES ?webdomains_to_search {" + unpacked_webdomain_string + "} \
                        {{?item wdt:P856 ?webdomains_to_search, ?found_domain.}} \
                        OPTIONAL {?item wdt:P452 ?industry.} \
                        OPTIONAL {?item wdt:P31 ?instanceOf.} \
                        BIND(IF(BOUND(?industry),?industry,?instanceOf) AS ?keyword). \
                        OPTIONAL {?item schema:description ?itemdescription. FILTER (lang(?itemdescription) = \"en\").} \
                        SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". } \
                    } \
                    ORDER BY ?itemLabel"

    sparql.setQuery(query_string)
    query_results = sparql.query().convert()
    results_df = pd.json_normalize(query_results['results']['bindings'])
    # Unfortunately this results contains columns we are not interested in e.g the types
    # This is because we set json as the return format but other formats are currently not working as intended
    # We will drop the columns we are not interested in
    results_df = results_df.loc[:, results_df.columns.str.contains('(value)')]
    # Get rid of the ".value" string behind our columns
    results_df.columns = results_df.columns.str.replace(".value", "")
    return results_df


if __name__ == '__main__':

    inflect_engine = inflect.engine()

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                           agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/106.0.0.0 Safari/537.36)")
    sparql.setReturnFormat(JSON)

    training_data = pd.read_csv("csv_files/training_data.csv")

    training_data["matches"] = ""

    httpwebdomain = training_data["httpwebdomain"].tolist()
    httpswebdomain = training_data["httpswebdomain"].tolist()
    joined_domains = list(zip(httpwebdomain, httpswebdomain))

    total_start = time.time()
    try:
        index = 0
        increaser = 85
        while index <= len(joined_domains):
            query_list = joined_domains[index:index + increaser]
            try:
                wiki_result_all = search_in_wikidata(query_list)
            except HTTPError as httpe:
                if httpe.code == 429:
                    retry_after = httpe.headers["Retry-After"]
                    if retry_after is not None:
                        try:
                            retry_after = int(retry_after)
                            print(f"Sleeping for {retry_after} seconds, Retry-After value was directly accessible")
                            time.sleep(retry_after)
                            wiki_result_all = search_in_wikidata(query_list)
                        except ValueError as ve:
                            retry_after = datetime.strptime(retry_after, '%a, %d %b %Y %H:%M:%S %Z')
                            current_time = datetime.now()
                            time_difference_in_seconds = (retry_after - current_time).total_seconds()
                            print(f"Sleeping for:{time_difference_in_seconds} seconds, Datetime-Object was used")
                            time.sleep(time_difference_in_seconds)
                            wiki_result_all = search_in_wikidata(query_list)
                else:
                    print("Something totally unexpected happened, please try again later")
                    print(httpe)
                    sys.exit(-1)
            tmp_index = index
            for element in joined_domains[index:index + increaser]:
                print(f"Currently at: {tmp_index}")
                wiki_result_df_http = wiki_result_all.loc[wiki_result_all["found_domain"] == element[0][1:-1]]
                wiki_result_df_https = wiki_result_all.loc[wiki_result_all["found_domain"] == element[1][1:-1]]
                wiki_result_df_joined = pd.concat([wiki_result_df_http, wiki_result_df_https], join="inner")
                wiki_results = tokenize_and_keep_only_nouns(wiki_result_df_joined)
                keywords_for_domain_guessing = create_list_from_dataframe_and_drop_duplicates(wiki_results)
                keywords_for_domain_guessing = remove_nan_and_empty_values_from_list(keywords_for_domain_guessing)
                keywords_for_domain_guessing += return_pluralized_nouns(keywords_for_domain_guessing)
                if len(keywords_for_domain_guessing) >= 1:
                    training_data.at[tmp_index, "matches"] = keywords_for_domain_guessing
                tmp_index += 1
            index += increaser
    finally:
        total_end = time.time()
        training_data = training_data[training_data["matches"] != ""]
        training_data.to_csv("csv_files/training_data.csv", index=False, mode='w')
        print(f"Took a total of {np.round(((total_end - total_start) / 60), 2)} minutes")
