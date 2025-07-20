# Data Preparation is done in AWS Glue job name "SubD_Churn_Data_Prep"

import pandas as pd
import numpy as np
import psycopg2
import datetime
from datetime import date, timedelta, datetime
import os
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
import sklearn.metrics
import imblearn
from imblearn.over_sampling import SMOTE
import pickle as cPickle
import boto3
import io

print("Packages IN!!")

connection = psycopg2.connect(host='**.*.**.***',
                         dbname='***********',
                         user='*****',
                         password='***************',
                        port='*****')
curs = connection.cursor();
print('Connection Done!!')

curs.execute('select * from nps.subdchurn_input_new_csv;')
data = curs.fetchall()
subd_input = pd.DataFrame(data,columns=["customercode","period","month","year","dtcode","increment","netamt","subd_increment","cat_range","sku_range","churn_flag","churn_risk_flag","l3m_declining_sale","l3m_0_sale","adjusted","increment_less_than_national","data_preparation_date"])
print(subd_input.head())
print(subd_input.shape)

subd_model = subd_input.copy()
subd_model['churn_risk_flag_new'] = np.where(subd_model['churn_risk_flag']=='nan',"Y",subd_model['churn_risk_flag'])
subd_model['l3m_0_sale_new'] = np.where(subd_model['l3m_0_sale']=='nan',"Y",subd_model['l3m_0_sale'])
subd_model['l3m_declining_sale_new'] = np.where(subd_model['l3m_declining_sale']=='nan',"Y",subd_model['l3m_declining_sale'])
subd_model['churn_flag_new'] = np.where(subd_model['churn_flag']=='nan',"Y",subd_model['churn_flag'])
print(subd_model.head())

churned = subd_model[subd_model['churn_risk_flag_new'] == "Y"]
print('percentage of churned customer: {}'.format(churned.shape[0]/subd_model.shape[0]))

not_churned = subd_model[subd_model['churn_risk_flag_new'] == "N"]
print('percentage of not-churned customer: {}'.format(not_churned.shape[0]/subd_model.shape[0]))

subd_model = subd_model.replace('nan',0)
subd_model = subd_model.replace(np.nan,0)
subd_model = subd_model.replace('null',0)
print(subd_model.head())
print('subd_model.shape')
print(subd_model.shape)

dt_details = subd_model.groupby(['customercode'],as_index=False).agg({'dtcode':'nunique'})
print('dt_details')
print(dt_details.head())

print('customer codes with more than one DTcodes mapped')
print(dt_details[dt_details['dtcode']>1]['customercode'].unique())

del subd_model['churn_risk_flag']
del subd_model['l3m_0_sale']
del subd_model['l3m_declining_sale']
del subd_model['churn_flag']

##################### Account Table : To map distributor Lat and Long #############################
curs.execute("select kunnr__c,geolocation__latitude__s,geolocation__longitude__s from std_sfdc_db.account where customer_group__c = 'DC' and Sales_Org_CEAT__c  = '1001' and kunnr__c !='null';")
data_acc = curs.fetchall()
account = pd.DataFrame(data_acc,columns=["dtcode","dt_latitude","dt_longitude"])
account = account.replace(np.nan,0)
account = account.replace('null',0)

account['dt_latitude'] = pd.to_numeric(account['dt_latitude'], errors='coerce')
account['dt_longitude'] = pd.to_numeric(account['dt_longitude'], errors='coerce')
account['dt_latitude'] = account['dt_latitude'].astype(float)
account['dt_longitude'] = account['dt_longitude'].astype(float)

print(account.head())

subd_model = pd.merge(left = subd_model, right = account, on = ['dtcode'], how = 'left')
subd_model = subd_model.replace(np.nan,0)
subd_model = subd_model.replace('null',0)
print(subd_model.head())
print('subd_model.shape after account table mapping')
print(subd_model.shape)

##################### Customer Master ######################

curs.execute("select kunnr__c,latitude_dms__c,longitude_dms__c from botree_sfdc.customer_master_report where source = 'SFDC';")
data_mast = curs.fetchall()
cust_master = pd.DataFrame(data_mast,columns=["customercode","latitude","longitude"])
cust_master = cust_master.replace(np.nan,0)
cust_master = cust_master.replace('null',0)

cust_master['latitude'] = pd.to_numeric(cust_master['latitude'], errors='coerce')
cust_master['longitude'] = pd.to_numeric(cust_master['longitude'], errors='coerce')
cust_master['latitude'] = cust_master['latitude'].astype(float)
cust_master['longitude'] = cust_master['longitude'].astype(float)

subd_model = pd.merge(left = subd_model, right = cust_master, on = ['customercode'], how = 'left')
subd_model = subd_model.replace(np.nan,0)
subd_model = subd_model.replace('null',0)
print(subd_model.head())
print('subd_model.shape after customer master mapping')
print(subd_model.shape)

subd_model['dt_latitude'] = subd_model['dt_latitude'].fillna(0)
subd_model['dt_latitude'] = subd_model['dt_latitude'].apply(lambda x: float(x))
subd_model['dt_longitude'] = subd_model['dt_longitude'].fillna(0)
subd_model['dt_longitude'] = subd_model['dt_longitude'].apply(lambda x: float(x))
subd_model['latitude'] = subd_model['latitude'].fillna(0)
subd_model['latitude'] = subd_model['latitude'].apply(lambda x: float(x))
subd_model['longitude'] = subd_model['longitude'].fillna(0)
subd_model['longitude'] = subd_model['longitude'].apply(lambda x: float(x))
print('subd_model.shape')
print(subd_model.shape)

dt_details_lat_long = subd_model.groupby(['customercode'],as_index=False).agg({'dt_latitude':'nunique','dt_longitude':'nunique','latitude':'nunique','longitude':'nunique'})
print('dt_details_lat_long')
print(dt_details_lat_long.head())

print('customer codes with more than one DTcodes mapped')
print(dt_details_lat_long[dt_details_lat_long['latitude']>1]['customercode'].unique())


######### https://gist.github.com/rochacbruno/2883505 #################

subd_model['s_lat'] = subd_model['dt_latitude']*np.pi/180.0
subd_model['s_lon'] = np.deg2rad(subd_model['dt_longitude']) 
subd_model['e_lat'] = np.deg2rad(subd_model['latitude'])
subd_model['e_lon'] = np.deg2rad(subd_model['longitude'])

# approximate radius of earth in km
R = 6373.0
subd_model['d'] = np.sin((subd_model['e_lat'] - subd_model['s_lat'])/2)**2 + np.cos(subd_model['s_lat'])*np.cos(subd_model['e_lat']) * np.sin((subd_model['e_lon'] - subd_model['s_lon'])/2)**2
subd_model['distance_km'] = 2 * R * np.arcsin(np.sqrt(subd_model['d']))
print(subd_model.head())
print(subd_model.columns)
print('subd_model')
print(subd_model.shape)

print('ord_enc = OrdinalEncoder')
ord_enc = OrdinalEncoder()
subd_model["churn_risk_code"] = ord_enc.fit_transform(subd_model[["churn_risk_flag_new"]])
subd_model["l3m_0_sale_code"] = ord_enc.fit_transform(subd_model[["l3m_0_sale_new"]])
subd_model["l3m_declining_sale_code"] = ord_enc.fit_transform(subd_model[["l3m_declining_sale_new"]])
subd_model["churn_flag_code"] = ord_enc.fit_transform(subd_model[["churn_flag_new"]])
subd_model["increment_less_than_national_code"] = ord_enc.fit_transform(subd_model[["increment_less_than_national"]])
subd_model["period_code"] = ord_enc.fit_transform(subd_model[["period"]])

subd_model_input = subd_model.copy()
print(subd_model_input.dtypes)

subd_model_input['netamt'] = subd_model_input['netamt'].astype(float)
subd_model_input['sku_range'] = subd_model_input['sku_range'].astype(int)
subd_model_input['cat_range'] = subd_model_input['cat_range'].astype(int)
subd_model_input['adjusted'] = subd_model_input['adjusted'].astype(float)
subd_model_input['increment'] = subd_model_input['increment'].astype(float)
subd_model_input['subd_increment'] = subd_model_input['subd_increment'].astype(float)

subd_model_input['subd_increment_1']=np.where(((subd_model_input['subd_increment'] == subd_model_input['subd_increment'].min()) | (subd_model_input['subd_increment'] == subd_model_input['subd_increment'].max())),0,subd_model_input['subd_increment'])
print(subd_model_input['subd_increment_1'].max())

subd_model_input['increment_1']=np.where(((subd_model_input['increment'] == subd_model_input['increment'].min()) | (subd_model_input['increment'] == subd_model_input['increment'].max())),0,subd_model_input['increment'])
print(subd_model_input['increment_1'].max())

print('Correlation with churn_flag_code')
print(subd_model_input.corr()['churn_flag_code'])

print('subd_model_input.columns')
print(subd_model_input.columns)
print('subd_model_input')
print(subd_model_input.shape)

subd_model_input['month'] = subd_model_input['month'].astype('int')
subd_model_input['year'] = subd_model_input['year'].astype('int')

predictors  = subd_model_input[['customercode','netamt','sku_range','adjusted','subd_increment_1','increment_1','month','year','period_code','cat_range','increment_less_than_national_code','l3m_declining_sale_code','l3m_0_sale_code','churn_risk_code','distance_km','longitude','latitude','dt_longitude','dt_latitude','data_preparation_date']]
print('predictors.columns')
print(predictors.columns)
print('target data')
target = subd_model_input[['churn_flag_code']]

X_train, X_test, y_train, y_test  =   train_test_split(predictors, target, test_size=.3,stratify=target)

print( "Predictor - Training : ", X_train.shape, "Predictor - Testing : ", X_test.shape )

print(X_train.head())

print('resampler To get the no of churn and non churn uniform for better modelling perfromance')
resampler = SMOTE(random_state=5) # To get the no of churn and non churn uniform for better modelling perfromance

train = X_train.copy()
del train['customercode']
del train['data_preparation_date']
del train['month']
del train['year']
print(train.shape)

test = X_test.copy()
del test['customercode']
del test['data_preparation_date']
del test['month']
del test['year']
print(test.shape)
print(X_test.head())

print('X_resampled, y_resampled')
X_resampled, y_resampled = resampler.fit_resample(train, y_train)
print(X_resampled.shape)
print(y_resampled.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
ct = ColumnTransformer(
    [('scaler', StandardScaler(), ['netamt','sku_range','adjusted','subd_increment_1','increment_1','period_code','cat_range','increment_less_than_national_code','l3m_declining_sale_code','l3m_0_sale_code','churn_risk_code','distance_km','longitude','latitude','dt_longitude','dt_latitude'])], remainder='passthrough')
X_scaled = ct.fit_transform(X_resampled)
X_test_scaled = ct.transform(test)

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
classifier=classifier.fit(X_scaled,y_resampled)

# save the classifier
with open('GNB_classifier.pkl', 'wb') as fid:
    cPickle.dump(classifier, fid)    

predictions=classifier.predict(X_test_scaled)

predictions= pd.DataFrame(predictions,columns=['Propensity_to_churn_code'])
print('predictions')
print(predictions.shape)
print('Propensity_to_churn_code')
print(predictions[predictions['Propensity_to_churn_code']==1].shape)

#Analyze accuracy of predictions
print(sklearn.metrics.confusion_matrix(y_test,predictions))

print(sklearn.metrics.accuracy_score(y_test, predictions))

# # # load it again
# with open('GNB_classifier.pkl', 'rb') as fid:
#     gnb_loaded = cPickle.load(fid)

today = str(date.today())
print('today')
print(today)
curr_year = int(today[:4])
# curr_year = str(today[:4])
print('curr_year')
print(curr_year)
curr_month = int(today[5:7])
# curr_month = str(today[5:7])
print('curr_month')
print(curr_month)

predictors_new = predictors[['netamt','sku_range','adjusted','subd_increment_1','increment_1','period_code','cat_range','increment_less_than_national_code','l3m_declining_sale_code','l3m_0_sale_code','churn_risk_code','distance_km','longitude','latitude','dt_longitude','dt_latitude']]
predictors_new = predictors_new.fillna(0)
predictors_new = predictors_new.replace('nan',0)
print('predictors_new')
print(predictors_new.shape)
print('predictors')
print(predictors.shape)
print('subd_model_input')
print(subd_model_input.shape)

X_scaled_full = ct.fit_transform(predictors_new)
print('X_scaled_full.shape')
print(X_scaled_full.shape)

predictions_full = classifier.predict(predictors_new)

predictions_full= pd.DataFrame(predictions_full,columns=['Propensity_to_churn_code'])
print('predictions_full')
print(predictions_full.shape)
predictions_full['Propensity_to_churn_model_output'] = np.where(predictions_full['Propensity_to_churn_code']==1,"Churn expected","Safe")
print(predictions_full[predictions_full['Propensity_to_churn_model_output']=='Churn expected'].shape)
print('predictors.shape')
print(predictors.shape)

output = pd.concat([predictors, predictions_full], axis=1)
print('output')
print(output.shape)
print(output.head())
# output = output[['customercode','Propensity_to_churn_model_output']]
print(output[output['Propensity_to_churn_model_output']=='Churn expected'].shape)

output['month'] = output['month'].astype('int')
output['year'] = output['year'].astype('int')
output_latest = output[(output['month']==curr_month) & (output['year']==curr_year)]
print('output_latest.shape')
print(output_latest.shape)
print(output_latest.head())
print(output_latest[output_latest['Propensity_to_churn_model_output']=='Churn expected'].shape)
del output_latest['Propensity_to_churn_code']
output_latest['output_generation_date'] = date.today()

dbuser = '******'
dbpassword = '*************'
dbhost = '**.*.**.***'
dbport = '*****'
dbname = '*******************'
s3_client = boto3.client('s3',aws_access_key_id= '*****************',aws_secret_access_key='*********************************')
s3_client.delete_object(Bucket = 'sub-dealer-churn-model', Key = "model_output.csv")
with io.StringIO() as csv_buffer:
    output_latest.to_csv(csv_buffer, index=False)
    response = s3_client.put_object(Bucket='sub-dealer-churn-model', Key="model_output.csv", Body=csv_buffer.getvalue())
print("Recommendations Written to S3")
curs.close()
connection.close()