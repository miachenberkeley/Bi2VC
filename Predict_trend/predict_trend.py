import numpy as np
from collections import Counter
import xlsxwriter
import pandas as pd
import re

workbook = xlsxwriter.Workbook('articles_tokens.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"key words")
worksheet.write(0,1,"frequency of the word")

'''



by_month = {}

for key, value in dictionnary.items():
    if len(value) != 0:
        all_tokens = [token for sublist in value for token in sublist]

        tampon = Counter(all_tokens).most_common(300)
        by_month[key] = tampon

print(by_month)'''


dictionnary = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/l_tokens.npy")
dictionnary = dictionnary[()]

all_tokens = [token for sublist in dictionnary for token in sublist]
#print(all_tokens)

t_counts = dict(Counter(all_tokens))
counter = 0
for key, value in sorted(t_counts.iteritems(), key=lambda (k,v): (v,k)): #sorted by key

    worksheet.write(counter, 0, key)
    worksheet.write(counter, 1, value)
    counter += 1
    print("%s: %s" % (key, value))



'''
A = pd.read_excel("/Users/chen/Desktop/Bi2VC/data/2017-04-06_09-53_json_extract.xlsx")
#print(type(A))

list_company = []
list= A["funded_organization properties - api_path"].values

for name in list:
    try:
        list_company = list_company + [name]
    except:
        print("saute", name)
        #list_company = list_company + [name]

#print(len(list_company))

unique = []
[unique.append(item) for item in list_company if item not in unique]


start_up_company = []
for i in unique:
    try:
        print(re.escape(i),i)
        start_up_company.append(i)
    except:
        print(i)





re_startups = re.compile(ur"({})".format(ur"|".join(re.escape(startup_name) for startup_name in start_up_company)))



dict = {}

for value in dictionnary:
    for tokens in value:
        try:
            a = re.search(re_startups,tokens)
            if a != "None":
                dict[a] += 1
                print(a,tokens)
            else:
                pass
        except:
            print("None", tokens)
print(dict)




#re_startups = re.compile(ur"({})".format(ur"|".join(re.escape(startup_name) for startup_name in list_company)))
#print(re_startups)

#print(list_company)

for i, j in A.iteritems():

    if i == "investor properties - web_path":
        for z in range(len(j)):
            a = j[z].replace("organization/", "")
            b = a.replace("-", " ")
            print(b)

    #print(i)

    if i == "funded_organization properties - api_path":

        print(j)
        for z in range(len(j)):

                name = j[1].replace("-", " ")
                #name = b.split()
                list_company.append(name)









'''





