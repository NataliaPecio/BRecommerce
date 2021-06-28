#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:12:11 2021

@author: natalka
"""
############################################################################### 1) Data preparation
############################################################################### 1.1) import libraries
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean, median, sin, cos, sqrt, arctan2, radians, std, select, percentile,corrcoef, round
from seaborn import boxplot
from statistics import mode
#
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
############################################################################### 1.2) read data
data = pd.read_csv("olist_orders_dataset.csv")
data = data.dropna()
# Dodać kolejne bazy, zmergować, pokazać średnie czasy dostawy (aktualne i estimated) 
# w zalezności od odległości między zip codem klienta i sprzedawcy
data_items = pd.read_csv("olist_order_items_dataset.csv")
data_sellers = pd.read_csv("olist_sellers_dataset.csv")
data_customers = pd.read_csv("olist_customers_dataset.csv")
data_geo = pd.read_csv("olist_geolocation_dataset.csv")

############################################################################### 1.3) merge tables
#zip code usunąć duplikaty
data_geo_u = data_geo.groupby('geolocation_zip_code_prefix')['geolocation_lat','geolocation_lng'].mean()
# geo zip code podział dla customer i seller
data_geo_c = data_geo_u
data_geo_s = data_geo_u
data_geo_c = data_geo_c.rename(columns = {'geolocation_lat':'cust_lat_1','geolocation_lng':'cust_lng_1'})
data_geo_s = data_geo_s.rename(columns = {'geolocation_lat':'sell_lat_1','geolocation_lng':'sell_lng_1'})
#Zamienić na radiany
data_geo_c['cust_lat']=radians(data_geo_c.cust_lat_1)
data_geo_c['cust_lng']=radians(data_geo_c.cust_lng_1)
data_geo_s['sell_lat']=radians(data_geo_s.sell_lat_1)
data_geo_s['sell_lng']=radians(data_geo_s.sell_lng_1)
# sprawdzenie powyżej                        

# zmergować data z customer
full_data = pd.merge(left=data, right=data_customers, left_on='customer_id', right_on='customer_id')
full_data = full_data.drop(["customer_unique_id"], axis=1)
# zmergować items z sellers - krok pośredni do sellers
items_sellers = pd.merge(left=data_items, right=data_sellers, left_on='seller_id', right_on='seller_id')
items_sellers = items_sellers.drop(["shipping_limit_date"], axis=1)
#usunąć dodatkowe itemy z zamówień - zostawić po jednym itemie dla zamówienia
items_sellers.drop(items_sellers[items_sellers['order_item_id']!=1].index, inplace=True)
#zmergować full data z sellers
full_data = pd.merge(left=full_data, right=items_sellers, left_on='order_id', right_on='order_id')
#duplikaty dla danej kolumny
#duplicate_order_id = items_sellers.groupby("order_id").filter(lambda x: len(x["seller_id"].unique()) > 1)
#zmergować full_data z zip_code
#dla customer
full_data = pd.merge(left=full_data, right=data_geo_c, how="left",left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')
#dla seller
full_data = pd.merge(left=full_data, right=data_geo_s, how="left",left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix')

full_data = full_data.dropna()
############################################################################### 1.4) calculate distance
full_data['dLat']=full_data['cust_lat']-full_data['sell_lat']
full_data['dLng']=full_data['cust_lng']-full_data['sell_lng']

R = 6373.0

#full_data['dLat'].astype(float)
#full_data['dLng'].astype(float)
#jak sprawdzić typ danych w kolumnie
#full_data['dLng'].dtype
# Haversine formula a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
full_data['a'] = sin(full_data.dLat/2)**2 + cos(full_data.cust_lat) * cos(full_data.sell_lat) * sin(full_data.dLng/2)**2
# c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
full_data['c'] = 2 * arctan2(sqrt(full_data.a), sqrt(1-full_data.a))

full_data['Distance'] = R*full_data.c
############################################################################## 1.5 Add distance levels
conditions = [
(full_data.Distance <= 500),
(full_data.Distance >500) & (full_data.Distance <= 1500),
(full_data.Distance > 1500) & (full_data.Distance <= 2500),
(full_data.Distance > 2500)]

values = [1,2,3,4]
                                  
full_data['Distance_level'] = select(conditions, values)    
                          
############################################################################### 1.6 Delete incorrect lat/lng
# lat
full_data.drop(full_data[(full_data['cust_lat_1']<-33.74987) | (full_data['cust_lat_1']>5.272222)].index, inplace=True)
full_data.drop(full_data[(full_data['sell_lat_1']<-33.74987) | (full_data['sell_lat_1']>5.272222)].index, inplace=True)
# lng
full_data.drop(full_data[(full_data['cust_lng_1']>-34.79308) | (full_data['cust_lng_1']<-73.98208)].index, inplace=True)
full_data.drop(full_data[(full_data['sell_lng_1']>-34.79308) | (full_data['sell_lng_1']<-73.98208)].index, inplace=True)
############################################################################### 2) Data exploration 
############################################################################### 2.1 delete non-delivered values
full_data.drop(full_data[full_data['order_status']!='delivered'].index, inplace=True)
############################################################################### 2.2 calculate delivery times in days
full_data["Del_act_time"]=(pd.to_datetime(full_data["order_delivered_customer_date"])- pd.to_datetime(full_data["order_purchase_timestamp"])).astype('timedelta64[D]')
# change 0 values to 1
full_data["Del_act_time"] = full_data["Del_act_time"].replace(0,1)
# Szacunkowy czas delivery - od zakupu (purchase) do daty oszacowanej dostawy (estimated delivery)
full_data["Del_est_time"]=(pd.to_datetime(full_data["order_estimated_delivery_date"])- pd.to_datetime(full_data["order_purchase_timestamp"])).astype('timedelta64[D]')
#Róznica między datą delivery a datą estimated metoda 1
full_data["Diff_act_est"]=(pd.to_datetime(full_data["order_estimated_delivery_date"])- pd.to_datetime(full_data["order_delivered_customer_date"])).astype('timedelta64[D]')
# Różnica między czasem delivery a czasem estimated delivery metoda 2
#pur_del_time["Diff_act_est_2"]=pur_del_time["Del_est_time"]-pur_del_time["Del_act_time"]
#pur_del_time["Diff_act_est_method_comparison"]=pur_del_time["Diff_act_est_2"]-pur_del_time["Diff_act_est"]
#max(pur_del_time["Diff_act_est_method_comparison"])
#type(pur_del_time.order_delivered_customer_date)
#pur_del_time.Diff_act_est.dtype

############################################################################### 2.3 Showing statistics
#plt.plot(full_data.Del_act_time, '.')
plt.hist(full_data.Del_act_time)
boxplot(full_data.Del_act_time)
print("min: ", round(min(full_data.Del_act_time)),
"max: ", round(max(full_data.Del_act_time)),
"mode: ", round(mode(full_data.Del_act_time)),
"avg: ", round(mean(full_data.Del_act_time)),
"standard deviation: ", round(std(full_data.Del_act_time)),
"median: ", round(median(full_data.Del_act_time)))
# Statystyki dla Del_est_time
plt.plot(full_data.Del_est_time, '.')
plt.hist(full_data.Del_est_time)
print("min: ", round(min(full_data.Del_est_time)),
"max: ", round(max(full_data.Del_est_time)),
"mode: ", round(mode(full_data.Del_est_time)),
"avg: ", round(mean(full_data.Del_est_time)),
"standard deviation: ", round(std(full_data.Del_est_time)),
"median: ", round(median(full_data.Del_est_time)))
################################################### Statystyki dla Diff_act_est
#plt.plot(full_data.Diff_act_est, '.')
#plt.hist(full_data.Diff_act_est)
plt.plot(full_data['Del_act_time'][::500], label='Delivery actual time')
plt.plot(full_data['Del_est_time'][::500], label='Delivery estimated time')
plt.legend(bbox_to_anchor=(1, 1), loc='lower right', borderaxespad=0.)
plt.show()
print("min: ", round(min(full_data.Diff_act_est)),
"max: ", round(max(full_data.Diff_act_est)),
"mode: ", round(mode(full_data.Diff_act_est)),
"avg: ", round(mean(full_data.Diff_act_est)),
"standard deviation: ", round(std(full_data.Diff_act_est)),
"median: ", round(median(full_data.Diff_act_est)))
############################################################################## 2.4 Outliers
############################################################################## Identifying outliers 
#Calculate Interquartile range
q25 = percentile(full_data.Del_act_time,25)
# q50 to inaczej mediana
q50 = percentile(full_data.Del_act_time,50) 
q75 = percentile(full_data.Del_act_time,75) 
iqr = q75 - q25

cut_off = iqr*1.5
upper_outliers = q75+cut_off

outliers = [x for x in full_data.Del_act_time if x > upper_outliers]
############################################################################## Exploring outliers 
outliers_base = full_data.copy()
outliers_base.drop(outliers_base[outliers_base.Del_act_time <= upper_outliers].index, inplace=True)
outliers_base["Del_to_carrier_time"]=(pd.to_datetime(full_data["order_delivered_carrier_date"])- pd.to_datetime(full_data["order_purchase_timestamp"])).astype('timedelta64[D]')
outliers_base["Del_to_carrier_time"] = outliers_base["Del_to_carrier_time"].replace(0,1)
outliers_base["Payment_time"]=(pd.to_datetime(full_data["order_approved_at"])- pd.to_datetime(full_data["order_purchase_timestamp"])).astype('timedelta64[D]')
outliers_base["Payment_time"] = outliers_base["Del_to_carrier_time"].replace(0,1)
outliers_base = outliers_base.reset_index()
#plot
plt.hist(outliers_base.Del_act_time, label="Delivery actual time (days)")
plt.legend(bbox_to_anchor=(1, 1), loc='lower right', borderaxespad=0.)
plt.show()
#plot - difference between del act time i del carrier rime
plt.plot(outliers_base['Payment_time'],'.', label='Payment time')
plt.plot(outliers_base['Del_act_time'], label='Delivery actual time')
plt.plot(outliers_base['Del_to_carrier_time'], label='Delivery to carrier time')
plt.legend(bbox_to_anchor=(1, 1), loc='lower right', borderaxespad=0.)
plt.show()
#
outliers_base.info
# percentage of outliers in database cut_off = 1.5*iqr
4986/95968*100
# percentage of outliers in database cut_off = 3*iqr
1306/95968*100
# database size without outliers cut_off = 1.5*iqr
95968-4986
# database size without outliers cut_off = 3*iqr
95968-1306
############################################################################### removing outliers from the database
full_data.drop(full_data[full_data.Del_act_time > upper_outliers].index, inplace=True)
############################################################################### Calculate distance levels
Levels_Time = full_data[['Distance_level','Del_act_time']]
Mean = round(Levels_Time.groupby('Distance_level')['Del_act_time'].mean())
Min = Levels_Time.groupby('Distance_level')['Del_act_time'].min()
Median = Levels_Time.groupby('Distance_level')['Del_act_time'].median()

print(Min)
print(Mean)
print(Median)

plt.plot(Mean,'o', label='Mean')
plt.plot(Min,'o', label='Min')
plt.plot(Median,'o', label='Median')
levs=[1,2,3,4]
my_xticks = ['<=500','500-1500','1500-2500','>2500']
plt.xticks(levs,my_xticks)
plt.yticks([2,4,6,8,10,12,14,16,18,20])
plt.xlabel('Distance levels (km)')
plt.ylabel('Delivery time (days)')
plt.legend(bbox_to_anchor=(1, 1), loc='lower right', borderaxespad=0.)
plt.grid(color='grey', linestyle='solid')
plt.show()

#Levels_Time_1 = Levels_Time.copy()
#Levels_Time_1.drop(Levels_Time_1[Levels_Time_1['Distance_level']!=1].index, inplace=True)
#Levels_Time_2 = Levels_Time.copy()
#Levels_Time_2.drop(Levels_Time_2[Levels_Time_2['Distance_level']!=2].index, inplace=True)
#Levels_Time_3 = Levels_Time.copy()
#Levels_Time_3.drop(Levels_Time_3[Levels_Time_3['Distance_level']!=3].index, inplace=True)
#Levels_Time_4 = Levels_Time.copy()
#Levels_Time_4.drop(Levels_Time_4[Levels_Time_4['Distance_level']!=4].index, inplace=True)

#q90_Level_1 = percentile(Levels_Time_1.Del_act_time,90) 
#q90_Level_2 = percentile(Levels_Time_2.Del_act_time,90) 
#q90_Level_3 = percentile(Levels_Time_3.Del_act_time,90) 
#q90_Level_4 = percentile(Levels_Time_4.Del_act_time,90) 


#plt.plot(1,q90_Level_1,'.', label='90% of values')
#plt.plot(Mean,'.', label='Mean')
#plt.plot(Min,'.', label='Min')
#plt.plot(Median,'.', label='Median')
#plt.legend(bbox_to_anchor=(1, 1), loc='lower right', borderaxespad=0.)
#plt.show()



############################################################################### MODEL

Model_full_data = full_data[['Distance','freight_value','Del_act_time']]

Model_full_data = round(Model_full_data)
X = Model_full_data[['Distance','freight_value']]
Y = Model_full_data['Del_act_time']
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=0)

#X_train = X_train.to_numpy()
#X_test = X_test.to_numpy()
#y_train = y_train.to_numpy()
#y_test = y_test.to_numpy()

#X_train = X_train.reshape(-1,1)
#X_test = X_test.reshape(-1,1)

#model = LogisticRegression(max_iter=10000)

#model = RandomForestClassifier(n_estimators=1000, random_state=0) 
#model.fit(X_train,y_train)

predictions = model.predict(X_test)

score = model.score(X_test, y_test)
print(score)

cm = classification_report(y_test, predictions)
print(cm)

plt.plot(X_test,y_test,'.')


plt.plot(X_test,predictions,'.')



plt.plot(X_train,y_train,'.')


plt.plot(full_data['seller_id'],full_data['Del_act_time'], '.')



print(corrcoef(full_data['Distance'],full_data['Del_act_time']))
print(corrcoef(full_data['freight_value'],full_data['Del_act_time']))
print(corrcoef(full_data['price'],full_data['Del_act_time']))
# medium positive correlation
model = SVC(kernel='linear')
model.fit(X_train,y_train)

#plt.plot(full_data.Distance,full_data.Del_act_time, '.')
#plt.plot(full_data.Distance_level,full_data.Del_act_time, '.')
#plt.plot(full_data.Distance, '.')
#print("min: ", round(min(full_data.Distance)),
#"max: ", round(max(full_data.Distance)),
#"avg: ", round(mean(full_data.Distance)),
#"standard deviation: ", round(std(full_data.Distance)),
#"median: ", round(median(full_data.Distance)))


############################# spróbować logistic regression z round(distance)
# OVERFITTING
# dodać więcej zmiennych 
