import os
import json
from functools import reduce
import operator
import re


# Example : print get_from_dict(json_dict, ['widget', 'text'])
def get_from_dict(data_dict, map_list):
    return reduce(operator.getitem, map_list, data_dict)


def json_extract(file_path):
    with open(file_path) as json_data:
        json_dict = json.load(json_data)
    return json_dict


def get_investors_name_condensed(json_dict):
    json_len = len(json_dict)
    investors_name = []

    for i in range(0, json_len):
        address = get_from_dict(json_dict, [str(i), 'data', 'paging', 'next_page_url'])
        name_search = re.search('organizations/(.+?)/investments', address)
        if name_search:
            name = name_search.group(1)
            investors_name.append(name)
            print name

    print investors_name


# Save only the names of the investors who have invested in at least one company
def get_investors_name(json_dict):
    json_len = len(json_dict)
    investors_name = []

    for i in range(0, json_len):
        temp1 = get_from_dict(json_dict, [str(i), 'data'])

        if 'items' in temp1:
            if len(temp1['items']) > 0:
                temp2 = get_from_dict(temp1['items'][0], ['relationships', 'investors'])
                if len(temp2) > 0:
                    name = get_from_dict(temp2[0], ['properties', 'name'])
                    investors_name.append(name)

        elif 'item' in temp1:
            if len(temp1) > 0:
                temp2 = get_from_dict(temp1, ['item', 'relationships', 'investors'])
                if len(temp2) > 0:
                    name = get_from_dict(temp2[0], ['properties', 'name'])
                    investors_name.append(name)

    return investors_name

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    json_dict = json_extract(dir_path + "/data/metadata_mia.json")

    investors_name = get_investors_name(json_dict)
    print investors_name


"""
create a list of functions that extract each main data of the datacrunch json file

json architecture:

--ARRAY--
    data // data on one investor (in the example here : zygote-venture) and its investments

        paging // get name of investor
            next_page_url : "https://api.crunchbase.com/v/3/organizations/zygote-ventures/investments?page=2" // give name of the investor

        items => --ARRAY-- // list of investments

            properties // of the investment made by the investor
                money_invested : null // I think always "null"
                money_invested_usd : null // I think always "null"
                money_invested_currency_code : "USD"
                is_lead_investor : false // only interesting info
                updated_at : 1453539877
                created_at : 1433702192

            relationships // info on the investment

                funding_round // info on the investment

                    relationships // info on the founded organization

                        funded_organization

                            properties
                                name : "Bolt Threads"
                                also_known_as ==> --ARRAY--

                                founded_on : "2009-08-01"
                                created_at : 1405983588 // date
                                updated_at : 1485726689 // date
                                closed_on : null // date (I guess)

                                founded_on_trust_code : 7
                                closed_on_trust_code : 0

                                permalink : "bolt-threads" // name
                                homepage_url : "http://www.boltthreads.com" // compagny address

                                num_employees_min : 11
                                num_employees_max : 50

                                stock_exchange : null
                                stock_symbol : null

                                number_of_investments : 0
                                total_funding_usd : 89999999

                                short_description : "Engineering '(...)'"
                                description : "We believe that '(..)'"
                    properties
                        announced_on : "2015-06-04"
                        created_at : 1433420308
                        updated_at : 1464908145
                        closed_on : null

                        web_path : "funding-round/a67902c71f21380746747f437c19381d" // to be combined with 'https://www.crunchbase.com/'

                        funding_type : "venture"

                        target_money_raised_currency_code : "USD"
                        target_money_raised_usd : null
                        target_money_raised : null

                        money_raised_currency_code : "USD"
                        money_raised : 32299999
                        money_raised_usd : 32299999

                        series : "B"

                        series_qualifier : null
                        announced_on_trust_code : 7 // number of investments (tbc)
                investors ==> array // only one "investor" every time because data concerns only one investor
                    type : "Organization"
                    properties
                        name : "Zygote Ventures"
                        also_known_as : null --ARRAY--
                        primary_role : "investor"
                        homepage_url : "http://www.zygoteventures.com"

                        role_investor : true
                        role_company : null
                        role_group : null
                        role_school : null

                        number_of_investments : 7
                        total_funding_usd : 0

                        short_description : "Zygote Ventures is a privately held seed/angel venture capital fund."
                        description : "Zygote Ventures is a privately held seed/angel venture capital fund. Zygote typically invests first, or very early, in innovative enterprises, most often technology, biotech, and agriculture.  As a privately held \u201cangel\u201d investor, Zygote Ventures falls somewhere between an entrepreneur and a traditional venture capital fund. Most VC\u2019s are limited partnerships, investing \u201cother people\u2019s money\u201d and earning much of their profit as fees and carry. In contrast, Zygote invests only its principal\u2019s money, and therefore earns no fees, profiting only from an increase in the value of the enterprise. This means that our interests are very closely aligned with the entrepreneur\u2019s. Without limited partners, Zygote can be less risk-averse than most VCs. Because there is no pressure to do a certain number of deals, or to invest a fixed-size fund, Zygote can choose to partner with those opportunities where we can provide real value. At the same time, as an angel investor

                        founded_on : "1981-01-01"
                        created_at : 1272106537
                        updated_at : 1478081659
                        is_closed : false
                        closed_on : null

                        founded_on_trust_code : 4
                        closed_on_trust_code : 0

                        num_employees_min : null
                        num_employees_max : null

                        stock_exchange : null
                        stock_symbol : null

"""

