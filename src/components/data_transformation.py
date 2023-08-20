import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import dist_calculate


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation stated')

            numerical_cols=['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
            'Restaurant_longitude', 'Delivery_location_latitude',
            'Delivery_location_longitude', 'Vehicle_condition',
            'multiple_deliveries', 'Time_taken (min)']

            categorical_cols=['Delivery_person_ID', 'Order_Date', 'Time_Orderd',
            'Time_Order_picked', 'Weather_conditions', 'Road_traffic_density',
            'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']

            # Define the custom ranking for each ordinal variable
            Type_of_order_cat = ['Drinks', 'Buffet', 'Snack', 'Meal']
            Type_of_vehicle_cat=['electric_scooter', 'scooter','bicycle', 'motorcycle']
            Road_traffic_density_cat=['Low','Medium','High','Jam']
            Weather_conditions_cat=['Sunny','Stormy','Sandstorms','Windy','Fog', 'Cloudy']
            City_cat=['Urban','Metropolitian','Semi-Urban']
            Festival_cat=['No','Yes']


            logging.info('Pipeline initiated')

            #Numerical pipeline

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            #Categorical pipeline

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=['Weather_conditions_cat', 'Road_traffic_density_cat',
                    'Type_of_order_cat', 'Type_of_vehicle_cat', 'City_cat','Festival_cat'])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

            logging.info('Pipeline completed')

        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Error in data transformation')

    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            #Calculate the distance between each pair of points

            train_df['distance']=np.nan
            test_df['distance']=np.nan

            
            for i in range(len(train_df)):
                train_df.loc[i,'distance']=dist_calculate(train_df.loc[i,'Restaurant_latitude'],
                                                    train_df.loc[i,'Restaurant_longitude'],
                                                    train_df.loc[i,'Delivery_location_latitude'],
                                                    train_df.loc[i,'Delivery_location_longitude']                             
                                                    )
                
            for i in range(len(test_df)):
                test_df.loc[i,'distance']=dist_calculate(test_df.loc[i,'Restaurant_latitude'],
                                                    test_df.loc[i,'Restaurant_longitude'],
                                                    test_df.loc[i,'Delivery_location_latitude'],
                                                    test_df.loc[i,'Delivery_location_longitude']                             
                                                    )
                
            train_df['time_diff']=(train_df['Time_Order_picked'].str.split(':').str[0].astype(float)*60+train_df['Time_Order_picked'].str.split(':').str[1].astype(float))-(train_df['Time_Orderd'].str.split(':').str[0].astype(float)*60+train_df['Time_Orderd'].str.split(':').str[1].astype(float))
            test_df['time_diff']=(test_df['Time_Order_picked'].str.split(':').str[0].astype(float)*60+test_df['Time_Order_picked'].str.split(':').str[1].astype(float))-(test_df['Time_Orderd'].str.split(':').str[0].astype(float)*60+test_df['Time_Orderd'].str.split(':').str[1].astype(float))
         
                
            logging.info('calculating distance completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name='Time_taken (min)'
            drop_columns=[target_column_name,'ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Delivery_person_ID','Order_Date','Time_Order_picked','Time_Orderd']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


            






