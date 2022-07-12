import csv
with open ('Cab_Data.csv','rb') as cab_data:
    reader = csv.reader(cab_data)
    p=[]
    y=[]
    cab[] = [row[3] for row in reader]
    for n in range(1,len(colume_cab)):
        if str(cab[n]) == 'Pink Cab' :
            p.append(n)
        elif str(cab[n]) == 'Yellow Cab' :
            y.append(n)
    ID = [row[1] for row in reader]
    date = [row[2] for row in reader]
    city = [row[4] for row in reader]
    km = [row[5] for row in reader]
    price = [row[6] for row in reader]
    cost = [roe[7] for row in reader]
    p_ID = ID[p]
    print p_ID

