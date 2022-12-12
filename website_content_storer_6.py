import asyncio
import time
import aiohttp
import aiofiles
import numpy as np
import pandas as pd


async def fetch(s, url, headers):
    try:
        filename = "http://www." + url
        async with s.get(filename, headers=headers) as resp:
            async with aiofiles.open("../website_responses/" + url, mode='w') as f:
                print(f"Beginning to write: {url}")
                await f.write(await resp.text())
                print(f"Finished writing to file: {url}")
    except Exception as e:
        print(e)
        return None


async def fetch_and_store_to_file(s, url, headers):
    await fetch(s, url, headers)


async def fetch_all(s, urls, headers):
    tasks = []
    for url in urls:
        tasks.append(fetch_and_store_to_file(s, url, headers))
    await asyncio.gather(*tasks)


async def main():
    headers = {'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/106.0.0.0 Safari/537.36)",
               "Accept-Language": "en-US"}
    urls = pd.read_csv("csv_files/training_data.csv")["Webdomains"].tolist()
    urls = np.array_split(urls, len(urls)/1000)
    for i in range(len(urls)):
        print(f"working on chunk: {i}")
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=30)
        session = aiohttp.ClientSession(timeout=timeout)
        await fetch_all(session, urls[i], headers)
        await session.close()

if __name__ == '__main__':
    start = time.time()

    asyncio.run(main())

    end = time.time()
    print(f"Took a total of {np.round(((end - start) / 60) / 60, 2)} hours")
