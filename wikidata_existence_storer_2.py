import time
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from urllib.error import HTTPError
from datetime import datetime


def trim_b_and_apostrophs(to_shorten):
    """Decoding the binary String will return a normal String"""
    return to_shorten.decode("utf-8")


def clean_webdomain(webdomain):
    if "https" in webdomain:
        return webdomain[12:-1]
    return webdomain[11:-1]


def get_ready_for_wikidata_search(webdomain):
    """Because the official addresses in the wikidata-pages use http as well as https we have to create a query for
    both cases """
    return "<http://www." + trim_b_and_apostrophs(webdomain) + "/>", \
           "<https://www." + trim_b_and_apostrophs(webdomain) + "/>"


def search_in_wikidata(list_to_query):
    unpacked_webdomain_string = ""
    for element in list_to_query:
        unpacked_webdomain_string += element[0] + " "
        unpacked_webdomain_string += element[1] + " "

    query_string = "SELECT ?item ?itemLabel ?found_webdomain " + \
                   "WHERE" + \
                   "{" + \
                   "VALUES ?webdomains_to_search {" + unpacked_webdomain_string + "}" + \
                   "?item wdt:P856 ?webdomains_to_search, ?found_webdomain." + \
                   "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }" + \
                   "}"

    print(f"First Webdomains:{list_to_query[:2]}...\n")
    sparql.setQuery(query_string)
    query_results = sparql.query().convert()
    results_df = pd.json_normalize(query_results['results']['bindings'])
    return results_df


def store_wikidata_information(domains_to_query, increaser=75, index=0):
    while index <= len(domains_to_query):
        tmp_query_list = domains_to_query[index:index + increaser]
        try:
            tmp_results = search_in_wikidata(tmp_query_list)
            tmp_results.to_csv("csv_files/training_data.csv", mode='a', header=False, index=False)
        except HTTPError as httpe:
            retry_after = httpe.headers["Retry-After"]
            if httpe.code == 429:
                try:
                    retry_after = int(retry_after)
                    print(f"Sleeping for {retry_after} seconds, Retry-After value was directly accessible")
                    time.sleep(retry_after)
                    tmp_results = search_in_wikidata(tmp_query_list)
                    tmp_results.to_csv("csv_files/training_data.csv", mode='a', header=False, index=False)
                except ValueError as ve:
                    retry_after = datetime.strptime(retry_after, '%a, %d %b %Y %H:%M:%S %Z')
                    current_time = datetime.now()
                    time_difference_in_seconds = (retry_after - current_time).total_seconds()
                    print(f"Sleeping for:{time_difference_in_seconds} seconds, Datetime Object was used")
                    time.sleep(time_difference_in_seconds)
                    tmp_results = search_in_wikidata(tmp_query_list)
                    tmp_results.to_csv("csv_files/training_data.csv", mode='a', header=False, index=False)
            else:
                print("Something totally unexpected happened... skipping over these domains...")
                print(httpe)
        index += increaser
        print(f"Currently at Index: {index}\n")


if __name__ == '__main__':
    start = time.time()

    data = pd.read_pickle("LLD-icon-full_data-names.pkl")
    query_list = [get_ready_for_wikidata_search(x) for x in data]
    data = data.reshape(data.size, -1)
    data = data.astype(str)
    data = pd.DataFrame(data)
    data.rename(columns={0: "Webdomains"}, inplace=True)
    webdomains = pd.DataFrame(query_list)
    webdomains.insert(2, column="Webdomains", value=data["Webdomains"])
    webdomains.rename(columns={
        0: "httpwebdomain",
        1: "httpswebdomain",
    }, inplace=True)

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                           agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/106.0.0.0 Safari/537.36)")
    sparql.setReturnFormat(JSON)

    store_wikidata_information(domains_to_query=query_list)

    wiki_results_df = pd.read_csv("csv_files/training_data.csv", header=None)
    wiki_results_df.rename(columns={3: "Webdomains"}, inplace=True)
    wiki_results_only_webdomains = pd.DataFrame(wiki_results_df['Webdomains'])
    wiki_results_only_webdomains.drop_duplicates(inplace=True)
    wiki_results_only_webdomains.rename(columns={0: "Webdomains"}, inplace=True)
    wiki_results_only_webdomains["Webdomains"] = wiki_results_only_webdomains["Webdomains"].apply(clean_webdomain)
    merged_df = pd.merge(wiki_results_only_webdomains, webdomains, on="Webdomains")
    merged_df.drop_duplicates(inplace=True)
    merged_df.to_csv("csv_files/training_data.csv", mode='w', index=False)

    end = time.time()
    print(f"Took a total of {np.round(((end - start) / 60), 2)} minutes")
