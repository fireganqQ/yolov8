import os

list = os.listdir(r"C:\Users\HUAWEI\Desktop\borsa yapay zeka\model-2\completion")

fileNameList = []
for i in list:
    if i.endswith('.txt'):
        fileNameList.append(i)

for i in fileNameList:
    os.system(
        fr"""curl --data-urlencode "markdown=$(cat 'C:\Users\HUAWEI\Desktop\borsa yapay zeka\model-2\completion\{i}')" --output {i.replace('.txt', '.pdf')} https://md-to-pdf.fly.dev"""
    )