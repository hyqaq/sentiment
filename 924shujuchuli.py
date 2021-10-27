import csv
with open("try.tsv","r",encoding='GBK') as f:
    data=csv.DictReader(f,delimiter='\t')
    mems=[mem for mem in data]
    review=[row["review"] for row in mems ]
    print(review)

