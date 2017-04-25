"""
Scrapes articles from TechCrunch by going through the articles on
http://techcrunch.com/page/<page number>

Saves the articles of each file into txt (the file name is extracted from the url)
A json file is created with each key being the file name and the value being a dict

Need to combine json files if the range is not all obtained at once.
"""


import re
import urllib2
import sys
from bs4 import BeautifulSoup
import pprint
import time
import json

# Change to what pages of TechCrunch need to be scraped
start_range = 1
end_range = 5000

meta_data_json = {}

for page_num in range(start_range, end_range+1):

    print "On page " + str(page_num)

    fo = open('log.txt', 'a')  # redirect all prints to this log file
    fo.write("On page " + str(page_num) + '\n')

    address = "http://techcrunch.com/tag/deadpool/page/" + str(page_num) + "/"
    html = urllib2.urlopen(address).read()
    soup = BeautifulSoup(html, 'lxml')
    results = soup.find_all("div", class_="block-content")

    for div_article in results:
        try:

            a_tag = div_article.find_all("a")

            # Valid article exists
            if len(a_tag) != 0:
                article_url = a_tag[0].get('href')

                # Check if it's a real article url
                # TechCrunch articles have year in them
                if "://techcrunch.com/2017" in article_url or "://techcrunch.com/2016" in article_url or "://techcrunch.com/2016" in article_url or "://techcrunch.com/2015" in article_url or "://techcrunch.com/2014" in article_url or "://techcrunch.com/2013" in article_url or "://techcrunch.com/2012" in article_url or "://techcrunch.com/2011" in article_url or "://techcrunch.com/2010" in article_url or "://techcrunch.com/2009" in article_url:
                    # Get article data
                    html = urllib2.urlopen(article_url).read()
                    soup = BeautifulSoup(html, 'lxml')

                    # Get main article text
                    main_article = soup.find_all("div", class_="article-entry text")[0]
                    article_txt = ""
                    for paragraph in main_article.find_all("p"):
                        article_txt += paragraph.getText().encode('ascii', errors='ignore')

                    # Save text to file
                    article_file = article_url.split("/")[6]
                    text_file = open("./scraped_files_cont/" + article_file + ".txt", "w")
                    text_file.write(article_txt)
                    text_file.close()

                    # Add metadata
                    date = soup.find("meta", {'name': 'sailthru.date'})['content']
                    title = soup.find("meta", {'name': 'sailthru.title'})['content']
                    author = soup.find("meta", {'class': "swiftype", "name": 'author'})['content']

                    related_links = []
                    for a_tag in main_article.find_all("a", href=True):
                        related_links.append(a_tag['href'])

                    meta_data_json[article_file] = {'title': title, 'date': date, 'url': article_url, 'author': author, 'related_links': related_links}

        except:  # Catches all errors
            print article_url + "\n"

            fo = open('log.txt', 'a')  # redirect all prints to this log file
            fo.write(article_url + "\n")

    # Save metadatajson in case program exits??
    if page_num % 100 == 0:
        with open("./scraped_files_third/metadata" + str(start_range) + ".json", "w") as fp:
            json.dump(meta_data_json, fp)
fo.close()

# Save metadatajson
with open("./scraped_files_third/metadata" + str(start_range) + ".json", "w") as fp:
    json.dump(meta_data_json, fp)
