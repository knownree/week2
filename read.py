import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt


def get_cab_data():
    cab_data = pd.read_csv('Cab_Data.csv')
    p=[]
    y=[]
    col3 = cab_data['Company']
    cab = np.array(col3)
    for n in range(1,len(cab_data)):#pink cap and yellow cap position in list
        if str(cab[n]) == 'Pink Cab' :
            p.append(n)
        elif str(cab[n]) == 'Yellow Cab' :
            y.append(n)
    col1 = cab_data['Transaction ID']
    tran = np.array(col1)#find value by position
    p_tran =tran[p]
    y_tran =tran[y]
    col2 = cab_data['Date of Travel']
    date = np.array(col2)
    p_date = date[p]
    y_date = date[y]
    col4 = cab_data['City']
    city = np.array(col4)
    p_city = city[p]
    y_city = city[y]    
    col5 = cab_data['KM Travelled']
    km = np.array(col5)
    p_km = km[p]
    y_km = km[y]
    col6 = cab_data['Price Charged']
    price = np.array(col6)
    col7 = cab_data['Cost of Trip']
    cost = np.array(col7)
    profit = price - cost
    p_profit = profit[p]
    y_profit = profit[y]
    prof_rate = profit/cost
    p_prof_rate = np.mean(prof_rate[p])
    y_prof_rate = np.mean(prof_rate[y])
    print('p:',p_prof_rate,'y:',y_prof_rate)
    return p_tran,y_tran,p_date,y_date,p_city,y_city,p_km,y_km,p_profit,y_profit,p,y

def date(p_date,y_date):
    p_date1=[]
    y_date1=[]
    for i in range(0,len(p_date)):#transfer to normal date
        a = xlrd.xldate_as_tuple(p_date[i],0)
        p_date1.append(a)
    for i in range(0,len(y_date)):
        a = xlrd.xldate_as_tuple(y_date[i],0)
        y_date1.append(a)
    p16=[]
    p17=[]
    p18=[]
    for i in range(0,len(p_date)):#seperate by year
        if p_date1[i][0] == 2016:
            p16.append(i)
        elif p_date1[i][0] == 2017:
            p17.append(i)
        elif p_date1[i][0] == 2018:
            p18.append(i)
    y16=[]
    y17=[]
    y18=[]
    for i in range(0,len(y_date)):
        if y_date1[i][0] == 2016:
            y16.append(i)
        elif y_date1[i][0] == 2017:
            y17.append(i)
        elif y_date1[i][0] == 2018:
            y18.append(i)
    psp=[]
    psu=[]
    pau=[]
    pwi=[]
    for i in range(0,len(p_date)):#seperate by season
        if p_date1[i][1] <4:
            psp.append(i)
        elif p_date1[i][1] >=4 and p_date1[i][1]<7:
            psu.append(i)
        elif p_date1[i][1] >=7 and p_date1[i][1]<10:
            pau.append(i)
        else:
            pwi.append(i)
    ysp=[]
    ysu=[]
    yau=[]
    ywi=[]
    for i in range(0,len(y_date)):
        if y_date1[i][1] <4:
            ysp.append(i)
        elif y_date1[i][1] >=4 and y_date1[i][1]<7:
            ysu.append(i)
        elif y_date1[i][1] >=7 and y_date1[i][1]<10:
            yau.append(i)
        else:
            ywi.append(i)
    return p16,p17,p18,y16,y17,y18,psp,psu,pau,pwi,ysp,ysu,yau,ywi

def get_tran_data(p_tran,y_tran,p,y,p16,p17,p18,y16,y17,y18,psp,psu,pau,pwi,ysp,ysu,yau,ywi):
    tran_data = pd.read_csv('Transaction_ID.csv')
    tran_data.sort_values('Transaction ID',ascending=True,inplace=True)
    col1 = tran_data['Transaction ID']
    tran = np.array(col1)
    col2 = tran_data['Customer ID']
    cust = np.array(col2)
    p_cust1 = cust[p]
    y_cust1 = cust[y]
    p16c1 = sorted(cust[p16])
    p17c1 = sorted(cust[p17])
    p18c1 = sorted(cust[p18])
    y16c1 = sorted(cust[y16])
    y17c1 = sorted(cust[y17])
    y18c1 = sorted(cust[y18])
    pspc1 = sorted(cust[psp])
    psuc1 = sorted(cust[psu])
    pauc1 = sorted(cust[pau])
    pwic1 = sorted(cust[pwi])
    yspc1 = sorted(cust[ysp])
    ysuc1 = sorted(cust[ysu])
    yauc1 = sorted(cust[yau])
    ywic1 = sorted(cust[ywi])
    datecust1=[p16c1,p17c1,p18c1,y16c1,y17c1,y18c1,pspc1,psuc1,pauc1,pwic1,yspc1,ysuc1,yauc1,ywic1]
    p_cust1.sort()
    y_cust1.sort()
    p_cust_num=[]
    y_cust_num=[]
    i=0;j=1
    for i in range(0,len(p_cust1)-1):#get each cap's cust number
        if p_cust1[i] == p_cust1[i+1]:
            j=j+1
        else:
            p_cust_num.append(j)
            j=1
    i=0;j=1
    for i in range(0,len(y_cust1)-1):
        if y_cust1[i] == y_cust1[i+1]:
            j=j+1
        else:
            y_cust_num.append(j)
            j=1
  
    i=0;j=0
    p_cust=[]
    y_cust=[]
    for i in range(len(p_cust_num)):
        p_cust.append(p_cust1[j])
        j=j+p_cust_num[i]
    i=0;j=0
    for i in range(len(y_cust_num)):
        y_cust.append(y_cust1[j])
        j=j+y_cust_num[i]
    return p_cust,y_cust,p_cust_num,y_cust_num,datecust1,cust

def get_cust_data(p_cust,y_cust,datecust1):
    cust_data=pd.read_csv('Customer_ID.csv')
    cust_data.sort_values('Customer ID', ascending=True,inplace=True)
    col1 = cust_data['Customer ID']
    cust = np.array(col1)
    col2 = cust_data['Gender']
    gend = np.array(col2)
    col3 = cust_data['Age']
    age = np.array(col3)
    col4 = cust_data['Income (USD/Month)']
    inc = np.array(col4)
    p=[]
    y=[]
    i=0
    j=0
    p_cust2=sorted(p_cust)
    while i<len(cust) and j<len(p_cust):#get position use the cust number
        if cust[i] == p_cust2[j]:
            p.append(i)
            i+=1
            j+=1
        elif cust[i] <p_cust2[j]:
            i+=1
        else:
            j+=1
    i=0
    j=0
    y_cust2=sorted(y_cust)
    while i<len(cust) and j<len(y_cust):
        if cust[i] == y_cust2[j]:
            y.append(i)
            i+=1
            j+=1
        elif cust[i] <y_cust2[j]:
            i+=1
        else:
            j+=1
    p16c=[];p17c=[];p18c=[];y16c=[];y17c=[];y18c=[];pspc=[];psuc=[];pauc=[];pwic=[];yspc=[];ysuc=[];yauc=[];ywic=[]
    datecust=[p16c,p17c,p18c,y16c,y17c,y18c,pspc,psuc,pauc,pwic,yspc,ysuc,yauc,ywic]#the cust number of each year each cap
    for i in range(0,13):
        j=0;k=0
        while j < len(cust) and k <len(datecust1[i]):
            if cust[j] == datecust1[i][k]:
                datecust[i].append(j)
                j+=1
                k+=1
            elif cust[j] < datecust1[i][k]:
                j+=1
            else:
                k+=1
    for i in range(0,len(gend)):
        if str(gend[i]) == 'Male':
            gend[i]=1
        else:
            gend[i]=0
    return gend,age,inc,datecust

def city(p_city,y_city):
    p_city.sort()
    y_city.sort()
    p_city_num=[]
    y_city_num=[]
    j=1
    for i in range(0,len(p_city)-1):#first sort it, and then find the number of same city and get the number of tran of a city
        if p_city[i+1] == p_city[i]:#also sort the csv file in excel to know the city order
            j+=1
        else:
            p_city_num.append(j)
            j=1
    p_city_num.append(j)
    j=1
    for i in range(0,len(y_city)-1):
        if y_city[i+1] == y_city[i]:
            j+=1
        else:
            y_city_num.append(j)
            j=1
    y_city_num.append(j)
    width=0.8
    yn=[0]*19
    for i in range(0,len(y_city_num)):
        yn[i] = p_city_num[i]+y_city_num[i]/2
    x=np.arange(19)
    for i in range(1,19):
        x[i]=2*i-1
    plt.figure(19,figsize=(20,5))
    plt.bar(x,p_city_num,width=width,color='pink',label='pink')
    plt.bar(x,y_city_num,width=width,color='y',bottom=p_city_num,label='yellow')
    xlabels=['NEW YORK NY','CHICAGO IL','LOS ANGELES CA','MIAMI FL','SILICON VALLEY','ORANGE COUNTY','SAN DIEGO CA','PHOENIX AZ','DALLAS TX','ATLANTA GA','DENVER CO','AUSTIN TX','SEATTLE WA','TUCSON AZ','SACRAMENTO CA','PITTSBURGH PA','WASHINGTON DC','NASHVILLE TN','BOSTON MA']  
    plt.xticks([i for i in x],xlabels,fontsize=7,rotation=45)
    for a,b in zip(x,p_city_num):
        plt.text(a,b/2,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x,yn):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'city')
    plt.ylabel(u'transaction amount')
    plt.title('transaction amout with city',loc='center')
    plt.savefig(fname='trancity.png',dpi=100)
    
    return p_city_num,y_city_num
    


def plotyear(p16,p17,p18,y16,y17,y18,p_profit,y_profit,p_tran,y_tran,p_km,y_km,gend,age,inc,datecust):
    p16p=sum(p_profit[p16])
    p17p=sum(p_profit[p17])
    p18p=sum(p_profit[p18])
    y16p=sum(y_profit[y16])
    y17p=sum(y_profit[y17])
    y18p=sum(y_profit[y18])
    p16t=len(p_tran[p16])
    p17t=len(p_tran[p17])
    p18t=len(p_tran[p18])
    y16t=len(y_tran[y16])
    y17t=len(y_tran[y17])
    y18t=len(y_tran[y18])
    p16k=sum(p_km[p16])
    p17k=sum(p_km[p17])
    p18k=sum(p_km[p18])
    y16k=sum(y_km[y16])
    y17k=sum(y_km[y17])
    y18k=sum(y_km[y18])
    p16g=gend[datecust[0]]
    p17g=gend[datecust[1]]
    p18g=gend[datecust[2]]
    y16g=gend[datecust[3]]
    y17g=gend[datecust[4]]
    y18g=gend[datecust[5]]
    p16a=age[datecust[0]]
    p17a=age[datecust[1]]
    p18a=age[datecust[2]]
    y16a=age[datecust[3]]
    y17a=age[datecust[4]]
    y18a=age[datecust[5]]
    p16i=inc[datecust[0]]
    p17i=inc[datecust[1]]
    p18i=inc[datecust[2]]
    y16i=inc[datecust[3]]
    y17i=inc[datecust[4]]
    y18i=inc[datecust[5]]
    p16i0=[];p16i3=[];p16i5=[];p17i0=[];p17i3=[];p17i5=[];p18i0=[];p18i3=[];p18i5=[]
    y16i0=[];y16i3=[];y16i5=[];y17i0=[];y17i3=[];y17i5=[];y18i0=[];y18i3=[];y18i5=[]
    datainc=[p16i0,p16i3,p16i5,p17i0,p17i3,p17i5,p18i0,p18i3,p18i5,y16i0,y16i3,y16i5,y17i0,y17i3,y17i5,y18i0,y18i3,y18i5]
    #cust of each year, each income stage, and each company
    for j in range(0,len(p16i)):
        if p16i[j]<3500:
            datainc[0].append(j)
        elif p16i[j]>=3500 and p16i[j]<15000:
            p16i3.append(j)
        else:
            p16i5.append(j)
    for j in range(0,len(p17i)):
        if p17i[j]<3500:
            p17i0.append(j)
        elif p17i[j]>=3500 and p17i[j]<15000:
            p17i3.append(j)
        else:
            p17i5.append(j)
    for j in range(0,len(p18i)):
        if p18i[j]<3500:
            p18i0.append(j)
        elif p18i[j]>=3500 and p18i[j]<15000:
            p18i3.append(j)
        else:
            p18i5.append(j)
    for j in range(0,len(y16i)):
        if y16i[j]<3500:
            y16i0.append(j)
        elif y16i[j]>=3500 and y16i[j]<15000:
            y16i3.append(j)
        else:
            y16i5.append(j)
    for j in range(0,len(y17i)):
        if y17i[j]<3500:
            y17i0.append(j)
        elif y17i[j]>=3500 and y17i[j]<15000:
            y17i3.append(j)
        else:
            y17i5.append(j)
    for j in range(0,len(y18i)):
        if y18i[j]<3500:
            y18i0.append(j)
        elif y18i[j]>=3500 and y18i[j]<15000:
            y18i3.append(j)
        else:
            y18i5.append(j)
    p16a18=[];p16a26=[];p16a41=[];p16a61=[];p17a18=[];p17a26=[];p17a41=[];p17a61=[];p18a18=[];p18a26=[];p18a41=[];p18a61=[]
    y16a18=[];y16a26=[];y16a41=[];y16a61=[];y17a18=[];y17a26=[];y17a41=[];y17a61=[];y18a18=[];y18a26=[];y18a41=[];y18a61=[]
    dataage=[p16a18,p16a26,p16a41,p16a61,p17a18,p17a26,p17a41,p17a61,p18a18,p18a26,p18a41,p18a61,y16a18,y16a26,y16a41,y16a61,y17a18,y17a26,y17a41,y17a61,y18a18,y18a26,y18a41,y18a61]
    #cust of each year, each age stage, each company
    for i in range(0,len(p16a)):
        if p16a[i]<26:
            p16a18.append(i)
        elif p16a[i]>=26 and p16a[i]<41:
            p16a26.append(i)
        elif p16a[i]>=41 and p16a[i]<60:
            p16a41.append(i)
        else:
            p16a61.append(i)
    for i in range(0,len(p17a)):
        if p17a[i]<26:
            p17a18.append(i)
        elif p17a[i]>=26 and p17a[i]<41:
            p17a26.append(i)
        elif p17a[i]>=41 and p17a[i]<60:
            p17a41.append(i)
        else:
            p17a61.append(i)
    for i in range(0,len(p18a)):
        if p18a[i]<26:
            p18a18.append(i)
        elif p18a[i]>=26 and p18a[i]<41:
            p18a26.append(i)
        elif p18a[i]>=41 and p18a[i]<60:
            p18a41.append(i)
        else:
            p18a61.append(i)
    for i in range(0,len(y16a)):
        if y16a[i]<26:
            y16a18.append(i)
        elif y16a[i]>=26 and y16a[i]<41:
            y16a26.append(i)
        elif y16a[i]>=41 and y16a[i]<60:
            y16a41.append(i)
        else:
            y16a61.append(i)
    for i in range(0,len(y17a)):
        if y17a[i]<26:
            y17a18.append(i)
        elif y17a[i]>=26 and y17a[i]<41:
            y17a26.append(i)
        elif y17a[i]>=41 and y17a[i]<60:
            y17a41.append(i)
        else:
            y17a61.append(i)
    for i in range(0,len(y18a)):
        if y18a[i]<26:
            y18a18.append(i)
        elif y18a[i]>=26 and y18a[i]<41:
            y18a26.append(i)
        elif y18a[i]>=41 and y18a[i]<60:
            y18a41.append(i)
        else:
            y18a61.append(i)
    #tran by years
    total_width=0.8
    n=2
    width=total_width/n
    x=np.arange(3)
    x=x-(total_width-width)/2
    y11=[p16t,p17t,p18t]
    y12=[y16t,y17t,y18t]
    plt.figure(1)
    plt.bar(x,y11,width=width,color='pink',label='pink')
    plt.bar(x+width,y12,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y11):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y12):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'transaction amount')
    plt.title('transaction amout with year',loc='center')
    plt.savefig(fname='tranyear.png',dpi=100)

    
    #profit by years
    y21=[p16p,p17p,p18p]
    y22=[y16p,y17p,y18p]
    plt.figure(2)
    plt.bar(x,y21,width=width,color='pink',label='pink')
    plt.bar(x+width,y22,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y21):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y22):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'total profit')
    plt.title('profit with year',loc='center')
    plt.savefig(fname='proyear.png',dpi=100)


    #km bu years
    y31=[p16k,p17k,p18k]
    y32=[y16k,y17k,y18k]
    plt.figure(3)
    plt.bar(x,y31,width=width,color='pink',label='pink')
    plt.bar(x+width,y32,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y31):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y32):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'distance visited/km')
    plt.title('distance visited with year',loc='center')
    plt.savefig(fname='kmyear.png',dpi=100)


    #km profit rate by years
    y41=np.array(y31)/np.array(y21)
    y42=np.array(y32)/np.array(y22)
    plt.figure(4)
    plt.bar(x,y41,width=width,color='pink',label='pink')
    plt.bar(x+width,y42,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y41):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y42):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'profit per km')
    plt.title('profit per km with year',loc='center')
    plt.savefig(fname='profkm.png',dpi=100)
    
    
    

    #gend with year
    y51=[sum(p16g),sum(p17g),sum(p18g)]#male
    y52=[sum(y16g),sum(y17g),sum(y18g)]
    y53=[len(p16g)-sum(p16g),len(p17g)-sum(p17g),len(p18g)-sum(p18g)]#famale
    y54=[len(y16g)-sum(y16g),len(y17g)-sum(y17g),len(y16g)-sum(y18g)]
    y55=[y51[0]/len(p16g),y51[1]/len(p17g),y51[2]/len(p18g)]
    y56=[y52[0]/len(y16g),y52[1]/len(y17g),y52[2]/len(y16g)]
    y57=[y53[0]/len(p16g),y53[1]/len(p17g),y53[2]/len(p18g)]
    y58=[y54[0]/len(y16g),y54[1]/len(y17g),y54[2]/len(y16g)]
    x1=np.arange(9)
    plt.figure(5)
    plt.bar(x,y51,width=width,color='pink',label='male')
    plt.bar(x+width,y52,width=width,color='y',label='male')
    plt.bar(x,y53,width=width,color='r',label='famale',bottom=y51)
    plt.bar(x+width,y54,width=width,color='blue',label='famale',bottom=y52)
    xlabels2=['2016','pink','yellow','2017','pink','yellow','2018','pink','yellow']
    plt.xticks([i/3-4*width/3 for i in x1],xlabels2)
    for a,b in zip(x,y51):
        plt.text(a,b/2,'%.f'%b,ha='center',va='center',fontsize=7)
    for a,b in zip(x+width,y52):
        plt.text(a,b/2,'%.f'%b,ha='center',va='center',fontsize=7)
    for i in range(0,3):
        plt.text(x[i],y51[i]+y53[i]/2,y53[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i]+width,y52[i]+y54[i]/2,y54[i],ha='center',va='center',fontsize=7)       
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'amount of customer')
    plt.title('custmer gender with year',loc='center')
    plt.savefig(fname='gendyear.png',dpi=100)

    #rate gend
    plt.figure(12)
    plt.bar(x,y55,width=width,color='pink',label='male')
    plt.bar(x+width,y56,width=width,color='y',label='male')
    plt.bar(x,y57,width=width,color='r',label='famale',bottom=y55)
    plt.bar(x+width,y58,width=width,color='blue',label='famale',bottom=y56)
    xlabels2=['2016','pink','yellow','2017','pink','yellow','2018','pink','yellow']
    plt.xticks([i/3-4*width/3 for i in x1],xlabels2)
    for a,b in zip(x,y55):
        plt.text(a,b/2,'%.3f'%b,ha='center',va='center',fontsize=7)
    for a,b in zip(x+width,y56):
        plt.text(a,b/2,'%.3f'%b,ha='center',va='center',fontsize=7)
    for i in range(0,3):
        plt.text(x[i],y55[i]+y57[i]/2,'%.3f'%y57[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i]+width,y56[i]+y58[i]/2,'%.3f'%y58[i],ha='center',va='center',fontsize=7)       
    plt.xlabel(u'year')
    plt.ylabel(u'arate of customer')
    plt.title('custmer gender with year',loc='center')
    plt.savefig(fname='rategendyear.png',dpi=100)


    #age with year
    y61=[len(p16a18),len(p17a18),len(p18a18)]
    y62=[len(y16a18),len(y17a18),len(y18a18)]
    y63=[len(p16a26),len(p17a26),len(p18a26)]
    y64=[len(y16a26),len(y17a26),len(y18a26)]
    y65=[len(p16a41),len(p17a41),len(p18a41)]
    y66=[len(y16a41),len(y17a41),len(y18a41)]
    y67=[len(p16a61),len(p17a61),len(p18a61)]
    y68=[len(y16a61),len(y17a61),len(y18a61)]
    y69=[0,0,0]
    y610=[0,0,0]
    y611=[0,0,0]
    y612=[0,0,0]
    for i in range(0,3):
        y69[i]=y61[i]+y63[i]
        y610[i]=y62[i]+y64[i]
        y611[i]=y61[i]+y63[i]+y65[i]
        y612[i]=y62[i]+y64[i]+y66[i]
    y613=[len(p16a18)+len(p16a26)+len(p16a41)+len(p16a61),len(p17a18)+len(p17a26)+len(p17a41)+len(p17a61),len(p18a18)+len(p18a26)+len(p18a41)+len(p18a61)]
    y614=[y62[0]+y64[0]+y66[0]+y68[0],y62[1]+y64[1]+y66[1]+y68[1],y62[2]+y64[2]+y66[2]+y68[2]]
    yy1=[y61,y63,y65,y67]
    y615=[0,0,0];y616=[0,0,0];y617=[0,0,0];y618=[0,0,0];y619=[0,0,0];y620=[0,0,0];y621=[0,0,0];y622=[0,0,0]
    yy2=[y615,y617,y619,y621]
    for i in range(0,4):
        for j in range(0,3):
            yy2[i][j]=yy1[i][j]/y613[j]
    yy3=[y62,y64,y66,y68]
    yy4=[y616,y618,y620,y622]
    for i in range(0,4):
        for j in range(0,3):
            yy4[i][j]=yy3[i][j]/y614[j]
    y623=[0,0,0]
    y624=[0,0,0]
    y625=[0,0,0]
    y626=[0,0,0]
    for i in range(0,3):
        y623[i]=y615[i]+y617[i]
        y624[i]=y616[i]+y618[i]
        y625[i]=y615[i]+y617[i]+y619[i]
        y626[i]=y616[i]+y618[i]+y620[i]

    x1=np.arange(9)
    plt.figure(6)
    plt.bar(x,y61,width=width,color='y',label='18-25')
    plt.bar(x+width,y62,width=width,color='y')
    plt.bar(x,y63,width=width,color='r',bottom=y61,label='26-40')
    plt.bar(x+width,y64,width=width,color='r',bottom=y62)
    for i in range(0,3):
        plt.bar(x[i],y65[i],width=width,color='g',bottom=y69[i],label='41-60')
        plt.bar(x[i]+width,y66[i],width=width,color='g',bottom=y610[i])
        plt.bar(x[i],y67[i],width=width,color='b',bottom=y611[i],label='>60')
        plt.bar(x[i]+width,y68[i],width=width,color='b',bottom=y612[i])
    xlabels2=['2016','pink','yellow','2017','pink','yellow','2018','pink','yellow']
    plt.xticks([i/3-4*width/3 for i in x1],xlabels2)
    for a,b in zip(x,y61):
        plt.text(a,b/2,'%.f'%b,ha='center',va='center',fontsize=7)
    for a,b in zip(x+width,y62):
        plt.text(a,b/2,'%.f'%b,ha='center',va='center',fontsize=7)
    for i in range(0,3):
        plt.text(x[i],y61[i]+y63[i]/2,y63[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i]+width,y62[i]+y64[i]/2,y64[i],ha='center',va='center',fontsize=7)    
        plt.text(x[i],y61[i]+y63[i]+y65[i]/2,y65[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i]+width,y62[i]+y64[i]+y66[i]/2,y66[i],ha='center',va='center',fontsize=7) 
        plt.text(x[i],y61[i]+y63[i]+y65[i]+y67[i]/2,y67[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i]+width,y62[i]+y64[i]+y66[1]+y68[i]/2,y66[i],ha='center',va='center',fontsize=7)  
    plt.legend(loc=2)  
    plt.xlabel(u'year')
    plt.ylabel(u'amount of age stage')
    plt.title('custmer age with year',loc='center')
    plt.savefig(fname='ageyear.png',dpi=100)

    #rate age
    plt.figure(13)
    plt.bar(x,y615,width=width,color='y',label='18-25')
    plt.bar(x,y617,width=width,color='r',bottom=y615,label='26-40')
    plt.bar(x,y619,width=width,color='g',bottom=y623,label='41-60')
    plt.bar(x,y621,width=width,color='b',bottom=y625,label='>60')
    xlabels2=['2016','pink','yellow','2017','pink','yellow','2018','pink','yellow']
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y615):
        plt.text(a,b/2,'%.3f'%b,ha='center',va='center',fontsize=7)
    for i in range(0,3):
        plt.text(x[i],y615[i]+y617[i]/2,'%.3f'%y617[i],ha='center',va='center',fontsize=7,)   
        plt.text(x[i],y615[i]+y617[i]+y619[i]/2,'%.3f'%y619[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i],y615[i]+y617[i]+y619[i]+y621[i]/2,'%.3f'%y621[i],ha='center',va='center',fontsize=7,)
    plt.xlabel(u'year')
    plt.ylabel(u'rate of age stage')
    plt.title('custmer age rate of pink company',loc='center')
    plt.savefig(fname='rateageyearp.png',dpi=100)
    

    #rate age2
    plt.figure(14)
    plt.bar(x,y616,width=width,color='y',label='18-25')
    plt.bar(x,y618,width=width,color='r',bottom=y616,label='26-40')
    plt.bar(x,y620,width=width,color='g',bottom=y624,label='41-60')
    plt.bar(x,y622,width=width,color='b',bottom=y626,label='>60')
    for a,b in zip(x,y616):
        plt.text(a,b/2,'%.3f'%b,ha='center',va='center',fontsize=7)
    for i in range(0,3):
        plt.text(x[i],y616[i]+y618[i]/2,'%.3f'%y618[i],ha='center',va='center',fontsize=7)    
        plt.text(x[i],y616[i]+y618[i]+y620[i]/2,'%.3f'%y620[i],ha='center',va='center',fontsize=7) 
        plt.text(x[i],y616[i]+y618[i]+y620[1]+y622[i]/2,'%.3f'%y622[i],ha='center',va='center',fontsize=7)  
    plt.xticks([i+width/2 for i in x],xlabels)
    plt.xlabel(u'year')
    plt.ylabel(u'rate of age stage')
    plt.title('custmer age rate of yellow company',loc='center')
    plt.savefig(fname='rateageyeary.png',dpi=100)
    

    

    #cust amount with year
    y71=[len(p16g),len(p17g),len(p18g)]
    y72=[len(y16g),len(y17g),len(y18g)]
    plt.figure(10)
    plt.bar(x,y71,width=width,color='pink',label='pink')
    plt.bar(x+width,y72,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y71):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y72):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend(loc=2)
    plt.xlabel(u'year')
    plt.ylabel(u'amount of customer')
    plt.title('customer amount with year',loc='center')
    plt.savefig(fname='custyear.png',dpi=100)

    #income with year
    y81=[len(p16i0),len(p17i0),len(p18i0)]
    y82=[len(y16i0),len(y17i0),len(y18i0)]
    y83=[len(p16i3),len(p17i3),len(p18i3)]
    y84=[len(y16i3),len(y17i3),len(y18i3)]
    y85=[len(p16i5),len(p17i5),len(p18i5)]
    y86=[len(y16i5),len(y17i5),len(y18i5)]
    y87=[0,0,0]
    y88=[0,0,0]
    for i in range(0,3):
        y87[i]=y81[i]+y83[i]
        y88[i]=y82[i]+y84[i]
    x1=np.arange(9)
    plt.figure(11)
    plt.bar(x,y81,width=width,color='y',label='under 3500')
    plt.bar(x+width,y82,width=width,color='y')
    plt.bar(x,y83,width=width,color='r',bottom=y81,label='3500-15000')
    plt.bar(x+width,y84,width=width,color='r',bottom=y82)
    plt.bar(x,y85,width=width,color='g',bottom=y87,label='>15000')
    plt.bar(x+width,y86,width=width,color='g',bottom=y88)
    xlabels2=['2016','pink','yellow','2017','pink','yellow','2018','pink','yellow']
    plt.xticks([i/3-4*width/3 for i in x1],xlabels2)
    for a,b in zip(x,y81):
        plt.text(a,b/2,'%.f'%b,ha='center',va='center',fontsize=7)
    for a,b in zip(x+width,y82):
        plt.text(a,b/2,'%.f'%b,ha='center',va='center',fontsize=7)
    for i in range(0,3):
        plt.text(x[i],y81[i]+y83[i]/2,y83[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i]+width,y82[i]+y84[i]/2,y84[i],ha='center',va='center',fontsize=7)    
        plt.text(x[i],y87[i]+y85[i]/2,y85[i],ha='center',va='center',fontsize=7,)
        plt.text(x[i]+width,y88[i]+y86[i]/2,y86[i],ha='center',va='center',fontsize=7) 
    plt.legend(loc=2)  
    plt.xlabel(u'year')
    plt.ylabel(u'amount of income stage')
    plt.title('income with year',loc='center')
    plt.savefig(fname='incyear.png',dpi=100)

    #income with age
    a18=sorted(p18a18+p16a18+p17a18+y16a18+y17a18+y18a18)
    a26=sorted(p16a26+p17a26+p18a26+y18a26+y17a26+y16a26)
    a41=sorted(p16a41+p17a41+p18a41+y16a41+y17a41+y18a41)
    a61=sorted(p16a61+p17a61+p18a61+y16a61+y17a61+y18a61)
    i0=sorted(p16i0+p17i0+p18i0+y16i0+y17i0+y18i0)
    i3=sorted(p16i3+p17i3+p18i3+y16i3+y17i3+y18i3)
    i5=sorted(p16i5+p17i5+p18i5+y16i5+y17i5+y18i5)
    a18i0=0;a18i3=0;a18i5=0;a26i0=0;a26i3=0;a26i5=0;a41i0=0;a41i3=0;a41i5=0;a61i0=0;a61i3=0;a61i5=0
    i=0;j=0
    while i<len(a18) and j<len(i0):#cust of each age stage and income stage
        if a18[i] == i0[j]:
            a18i0+=1
            i+=1
            j+=1
        elif a18[i]<i0[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a18) and j<len(i3):
        if a18[i] == i3[j]:
            a18i3+=1
            i+=1
            j+=1
        elif a18[i]<i3[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a18) and j<len(i5):
        if a18[i] == i5[j]:
            a18i5+=1
            i+=1
            j+=1
        elif a18[i]<i5[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a26) and j<len(i0):
        if a26[i] == i0[j]:
            a26i0+=1
            i+=1
            j+=1
        elif a26[i]<i0[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a26) and j<len(i3):
        if a26[i] == i3[j]:
            a26i3+=1
            i+=1
            j+=1
        elif a26[i]<i3[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a26) and j<len(i5):
        if a26[i] == i5[j]:
            a26i5+=1
            i+=1
            j+=1
        elif a26[i]<i5[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a41) and j<len(i0):
        if a41[i] == i0[j]:
            a41i0+=1
            i+=1
            j+=1
        elif a41[i]<i0[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a41) and j<len(i3):
        if a41[i] == i3[j]:
            a41i3+=1
            i+=1
            j+=1
        elif a41[i]<i3[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a41) and j<len(i5):
        if a41[i] == i5[j]:
            a41i5+=1
            i+=1
            j+=1
        elif a41[i]<i5[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a61) and j<len(i0):
        if a61[i] == i0[j]:
            a61i0+=1
            i+=1
            j+=1
        elif a61[i]<i0[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a61) and j<len(i3):
        if a61[i] == i3[j]:
            a61i3+=1
            i+=1
            j+=1
        elif a61[i]<i3[j]:
            i+=1
        else:
            j+=1
    i=0;j=0
    while i<len(a61) and j<len(i5):
        if a61[i] == i5[j]:
            a61i5+=1
            i+=1
            j+=1
        elif a61[i]<i5[j]:
            i+=1
        else:
            j+=1
    y91=[a18i0,a18i3,a18i5]
    y92=[a26i0,a26i3,a26i5]
    y93=[a41i0,a41i3,a41i5]
    y94=[a61i0,a61i3,a61i5]
    y95=[0,0,0]
    y96=[0,0,0]
    for i in range(0,3):
        y95[i]=y91[i]+y92[i]
        y96[i]=y91[i]+y92[i]+y93[i]
    plt.figure(15)
    plt.bar(x,y91,width=width,color='y',label='18-25')
    plt.bar(x,y92,width=width,color='r',bottom=y91,label='26-40')
    plt.bar(x,y93,width=width,color='g',bottom=y95,label='41-60')
    plt.bar(x,y94,width=width,color='b',bottom=y96,label='>60')
    for a,b in zip(x,y91):
        plt.text(a,b/2,'%.3f'%b,ha='center',va='center',fontsize=7)
    for i in range(0,3):
        plt.text(x[i],y91[i]+y92[i]/2,'%.3f'%y92[i],ha='center',va='center',fontsize=7)    
        plt.text(x[i],y91[i]+y92[i]+y93[i]/2,'%.3f'%y93[i],ha='center',va='center',fontsize=7) 
        plt.text(x[i],y91[i]+y92[i]+y93[i]+y94[i]/2,'%.3f'%y94[i],ha='center',va='center',fontsize=7)  
    xlabels1=['<3500','3500-15000','>15000']
    plt.legend()
    plt.xticks([i+width/2 for i in x],xlabels1)
    plt.xlabel(u'income')
    plt.ylabel(u'amount of customers')
    plt.title('age atage with income stage',loc='center')
    plt.savefig(fname='ageinc.png',dpi=100)

    #tran per cust with year
    y101=[25079/15729,30321/17785,29310/17598]
    y102=[82239/28345,98189/30634,94253/30748]
    plt.figure(16)
    plt.bar(x,y101,width=width,color='pink',label='pink')
    plt.bar(x+width,y102,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y101):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y102):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'tran per customer')
    plt.title('transaction per customer with year',loc='center')
    plt.savefig(fname='trancust.png',dpi=100)

    #profit per cust with year
    y111=[0,0,0]
    y112=[0,0,0]
    for i in range(0,3):
        y111[i]=y21[i]/y71[i]
        y112[i]=y22[i]/y72[i]
    plt.figure(17)
    plt.bar(x,y111,width=width,color='pink',label='pink')
    plt.bar(x+width,y112,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y111):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y112):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'prof per customer')
    plt.title('profit per customer with year',loc='center')
    plt.savefig(fname='profcust.png',dpi=100)

    #prof per tran
    y121=[0,0,0]
    y122=[0,0,0]
    for i in range(0,3):
        y121[i]=y21[i]/y11[i]
        y122[i]=y22[i]/y12[i]
    plt.figure(18)
    plt.bar(x,y121,width=width,color='pink',label='pink')
    plt.bar(x+width,y122,width=width,color='y',label='yellow')
    xlabels=['2016','2017','2018']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y121):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y122):
        plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'year')
    plt.ylabel(u'prof per tran')
    plt.title('profit per transaction with year',loc='center')
    plt.savefig(fname='proftran.png',dpi=100)



    datainc1=[p16i0+p17i0+p18i0+y16i0+y17i0+y18i0,p16i3+p17i3+p18i3+y16i3+y17i3+y18i3,p16i5+p17i5+p18i5+y16i5+y17i5+y18i5]
    dataage1=[p16a18+p17a18+p18a18+y16a18+y17a18+y18a18,p16a26+p17a26+p18a26+y16a26+y17a26+y18a26,p16a41+p17a41+p18a41+y16a41+y17a41+y18a41,p16a61+p17a61+p18a61+y16a61+y17a61+y18a61]
    return dataage1,datainc1

def transfer(dataage1,datainc1):# the income position and age position is under cust number, tranfer it to transaction ID, so it can be connected with tran and profit
    data = pd.read_csv('Transaction_ID.csv')
    data.sort_values('Customer ID',ascending=True,inplace=True)
    col1= data['Transaction ID']
    col2=data['Customer ID']
    tran=np.array(col1)
    cust=np.array(col2)
    a18c=[];a26c=[];a41c=[];a61c=[]
    dataaget=[a18c,a26c,a41c,a61c]
    i0c=[];i3c=[];i5c=[]
    datainct=[i0c,i3c,i5c]
    for i in range(0,len(dataage1)):
        j=0;k=0
        while j<len(cust) and k<len(dataage1[i]):
            if cust[j] == dataage1[i][k]:
                dataaget[i].append(j)
                j+=1
            elif cust[j]>dataage1[i][k]:
                k+=1
            else:
                j+=1
    for i in range(0,len(datainc1)):
        j=0;k=0
        while j<len(cust) and k<len(datainc1[i]):
            if cust[j] == datainc1[i][k]:
                datainct[i].append(j)
                j+=1
            elif cust[j]>datainc1[i][k]:
                k+=1
            else:
                j+=1
    a18=[];a26=[];a41=[];a61=[]
    i0=[];i3=[];i5=[]
    a18=sorted(tran[a18c])
    a26=sorted(tran[a26c])
    a41=sorted(tran[a41c])
    a61=sorted(tran[a61c])
    i0=sorted(tran[i0c])
    i3=sorted(tran[i3c])
    i5=sorted(tran[i5c])
    dataaget1=[a18,a26,a41,a61]
    datainct1=[i0,i3,i5]
    return dataaget1,datainct1


def mixplot(p_tran,y_tran,p_profit,y_profit,dataaget1,datainct1):
    p_tran.sort()
    y_tran.sort()
    pta18=[];pta26=[];pta41=[];pta61=[];yta18=[];yta26=[];yta41=[];yta61=[]
    agept=[pta18,pta26,pta41,pta61]
    ageyt=[yta18,yta26,yta41,yta61]
    pti0=[];pti3=[];pti5=[];yti0=[];yti3=[];yti5=[]
    incpt=[pti0,pti3,pti5]
    incyt=[yti0,yti3,yti5]
    for i in range(0,len(dataaget1)):
        j=0;k=0
        while j<len(p_tran) and k<len(dataaget1[i]):
            if p_tran[j] == dataaget1[i][k]:
                agept[i].append(j)
                j+=1
                k+=1
            elif p_tran[j]>dataaget1[i][k]:
                k+=1
            else:
                j+=1
        j=0;k=0
        while j<len(y_tran) and k<len(dataaget1[i]):
            if y_tran[j] == dataaget1[i][k]:
                ageyt[i].append(j)
                j+=1
                k+=1
            elif y_tran[j]>dataaget1[i][k]:
                k+=1
            else:
                j+=1
    for i in range(0,len(datainct1)):
        j=0;k=0
        while j<len(p_tran) and k<len(datainct1[i]):
            if p_tran[j] == datainct1[i][k]:
                incpt[i].append(j)
                j+=1
                k+=1
            elif p_tran[j]>datainct1[i][k]:
                k+=1
            else:
                j+=1
        j=0;k=0
        while j<len(y_tran) and k<len(datainct1[i]):
            if y_tran[j] == datainct1[i][k]:
                incyt[i].append(j)
                j+=1
            elif y_tran[j]>datainct1[i][k]:
                k+=1
            else:
                j+=1
    ppa18=p_profit[pta18]
    ppa26=p_profit[pta26]
    ppa41=p_profit[pta41]
    ppa61=p_profit[pta61]
    ppi0=p_profit[pti0]
    ppi3=p_profit[pti3]
    ppi5=p_profit[pti5]
    ypa18=y_profit[yta18]
    ypa26=y_profit[yta26]
    ypa41=y_profit[yta41]
    ypa61=y_profit[yta61]
    ypi0=y_profit[yti0]
    ypi3=y_profit[yti3]
    ypi5=y_profit[yti5]

    #tran with age
    total_width=0.8
    n=2
    width=total_width/n
    x=np.arange(4)
    x=x-(total_width-width)/2
    y11=[len(pta18),len(pta26),len(pta41),len(pta61)]
    y12=[len(yta18),len(yta26),len(yta41),len(yta61)]
    plt.figure(20)
    plt.bar(x,y11,width=width,color='pink',label='pink')
    plt.bar(x+width,y12,width=width,color='y',label='yellow')
    xlabels=['18-25','26-40','41-60','>60']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y11):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y12):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'age group')
    plt.ylabel(u'transaction amount')
    plt.title('transaction amout with age',loc='center')
    plt.savefig(fname='tranage.png',dpi=100)

    #tran with inc
    x=np.arange(3)
    x=x-(total_width-width)/2
    y21=[len(pti0),len(pti3),len(pti5)]
    y22=[len(yti0),len(yti3),len(yti5)]
    plt.figure(21)
    plt.bar(x,y21,width=width,color='pink',label='pink')
    plt.bar(x+width,y22,width=width,color='y',label='yellow')
    xlabels=['<3500','3500-15000','>15000']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y21):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y22):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'income group')
    plt.ylabel(u'transaction amount')
    plt.title('transaction amout with income',loc='center')
    plt.savefig(fname='traninc.png',dpi=100)

    #profit with age
    x=np.arange(4)
    x=x-(total_width-width)/2
    y31=[np.mean(ppa18),np.mean(ppa26),np.mean(ppa41),np.mean(ppa61)]
    y32=[np.mean(ypa18),np.mean(ypa26),np.mean(ypa41),np.mean(ypa61)]
    plt.figure(22)
    plt.bar(x,y31,width=width,color='pink',label='pink')
    plt.bar(x+width,y32,width=width,color='y',label='yellow')
    xlabels=['18-25','26-40','41-60','>60']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y31):
        plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y32):
        plt.text(a,b/2,'%.2f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'age group')
    plt.ylabel(u'average profit')
    plt.title('average profit with age',loc='center')
    plt.savefig(fname='profage.png',dpi=100)

    #profit with inc
    x=np.arange(3)
    x=x-(total_width-width)/2
    y41=[np.mean(ppi0),np.mean(ppi3),np.mean(ppi5)]
    y42=[np.mean(ypi0),np.mean(ypi3),np.mean(ypi5)]
    plt.figure(23)
    plt.bar(x,y41,width=width,color='pink',label='pink')
    plt.bar(x+width,y42,width=width,color='y',label='yellow')
    xlabels=['<3500','3500-15000','>15000']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y41):
        plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y42):
        plt.text(a,b/2,'%.2f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'income group')
    plt.ylabel(u'average profit')
    plt.title('average profit with income',loc='center')
    plt.savefig(fname='profinc.png',dpi=100)

    
def plotmonth(p_tran,y_tran,p_km,y_km,p_profit,y_profit,psp,psu,pau,pwi,ysp,ysu,yau,ywi):
    pspp=sum(p_profit[psp])
    psup=sum(p_profit[psu])
    paup=sum(p_profit[pau])
    pwip=sum(p_profit[pwi])
    yspp=sum(y_profit[ysp])
    ysup=sum(y_profit[ysu])
    yaup=sum(y_profit[yau])
    ywip=sum(y_profit[ywi]) 
    pspt=len(p_tran[psp])
    psut=len(p_tran[psu])
    paut=len(p_tran[pau])
    pwit=len(p_tran[pwi])
    yspt=len(y_tran[ysp])
    ysut=len(y_tran[ysu])
    yaut=len(y_tran[yau])
    ywit=len(y_tran[ywi])   
    pspk=sum(p_km[psp])
    psuk=sum(p_km[psu])
    pauk=sum(p_km[pau])
    pwik=sum(p_km[pwi])
    yspk=sum(y_km[ysp])
    ysuk=sum(y_km[ysu])
    yauk=sum(y_km[yau])
    ywik=sum(y_km[ywi])

    #profit with season
    total_width=0.8
    n=2
    width=total_width/n
    x=np.arange(4)
    x=x-(total_width-width)/2
    y11=[pspp,psup,paup,pwip]
    y12=[yspp,ysup,yaup,ywip]
    plt.figure(24)
    plt.bar(x,y11,width=width,color='pink',label='pink')
    plt.bar(x+width,y12,width=width,color='y',label='yellow')
    xlabels=['spring','summer','autumn','winter']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y11):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y12):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'seasons')
    plt.ylabel(u'total profit')
    plt.title('profit with seasons',loc='center')
    plt.savefig(fname='profsea.png',dpi=100)

    #tran with season
    y21=[pspt,psut,paut,pwit]
    y22=[yspt,ysut,yaut,ywit]
    plt.figure(25)
    plt.bar(x,y21,width=width,color='pink',label='pink')
    plt.bar(x+width,y22,width=width,color='y',label='yellow')
    xlabels=['spring','summer','autumn','winter']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y21):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y22):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'seasons')
    plt.ylabel(u'transaction amount')
    plt.title('transaction amount with seasons',loc='center')
    plt.savefig(fname='transea.png',dpi=100)

    #km with seasons
    y31=[pspk,psuk,pauk,pwik]
    y32=[yspk,ysuk,yauk,ywik]
    plt.figure(26)
    plt.bar(x,y31,width=width,color='pink',label='pink')
    plt.bar(x+width,y32,width=width,color='y',label='yellow')
    xlabels=['spring','summer','autumn','winter']  
    plt.xticks([i+width/2 for i in x],xlabels)
    for a,b in zip(x,y31):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(x+width,y32):
        plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=7)
    plt.legend()
    plt.xlabel(u'seasons')
    plt.ylabel(u'total distance')
    plt.title('distance with seasons',loc='center')
    plt.savefig(fname='kmsea.png',dpi=100)
    
    




def main():
    a=get_cab_data()
    b=date(a[2],a[3])
    c=get_tran_data(a[0],a[1],a[10],a[11],b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13])
    d=get_cust_data(c[0],c[1],c[4])
    e=city(a[4],a[5])
    f=plotyear(b[0],b[1],b[2],b[3],b[4],b[5],a[8],a[9],a[0],a[1],a[6],a[7],d[0],d[1],d[2],d[3])
    g=transfer(f[0],f[1])
    h=mixplot(a[0],a[1],a[8],a[9],g[0],g[1])
    i=plotmonth(a[0],a[1],a[6],a[7],a[8],a[9],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13])
    print('finish')

if __name__=='__main__':
    main()
    


   


