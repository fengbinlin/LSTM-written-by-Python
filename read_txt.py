f=open('d:/Users/B0/Desktop/数据集.txt', encoding='utf-8')
txt=[]
for line in f:
    line=line.replace('\n', '')
    txt.append(str(line.strip("1,")))
print(txt)