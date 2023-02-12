import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st 
from functions import find_target #Function for finding target
from functions import analyse_catagorical_col  #function for analyzing ane categorical column that we want 
from functions import heat_map_coor_plot #Function to perform correlation heatmap analysis 
from functions import plot_null_values #Function to check null values
from functions import replace_yes_no #Function to replace irregularities in data 
from functions import rep_null_val #Function to replace null values
from functions import drop_columns #Function to drop columns 
from functions import cleaning_pipeline #Function that includes cleaning replacing null values and dropping unnecessary columns
from functions import data_dictionary # variable dictionary 
from functions import problem_stat # problem statement string 
from functions import steps # Steps to be performed 
from functions import target_inference# target variable inference
from functions import check_all_member_same_target # as the name suggests it check how many families does not have same target for all variable
from functions import check_with_head_or_not # checks for how many families are without heads
################################################################################################################################
#functions
# To Improve speed and cache data
@st.cache_data(persist=True) 

def Read_data(filename):
    """Function reads csv data and returns as a dataframe"""
    return pd.read_csv(filename)

################################################################################################################################
#reading data     
income_train_df=Read_data('train.csv')
income_test_df=Read_data('test.csv')
###############################################################################################################################
#Separating columns in different data types 
float_columns=income_train_df.select_dtypes('float').columns.tolist()
int_columns=income_train_df.select_dtypes('int').columns.tolist()
object_columns=income_train_df.select_dtypes('object').columns.tolist()
###############################################################################################################################
# Title and Subheader
def main ():# beginning of the statement execution
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Qualifications </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("EDA Web App")
    st.text("Team: Ashish Sarkar, Prajwal Rajput,Shraddha Mungal,Aryan Patle")
    st.text("Mentor: Shriti Dutta")
    if st.sidebar.checkbox('Problem Statement'):
        problem_stat
    if st.sidebar.checkbox('Show Data Dictionary'):
        data_dictionary
    if st.sidebar.checkbox('Steps To be performed'):
         steps 
    if st.sidebar.checkbox('Finding Target Variable and Checking for Bias'):
        Target=find_target(income_train_df,income_test_df)
        st.text(f"The {find_target(income_train_df,income_test_df)} is the Output Variable")
        st.subheader(f"{Target} variable cardinal distribution")
        st.write((income_train_df[Target].value_counts()/len(income_train_df))*100)
        st.subheader("Mappings")
        """  
             * 1 : Extreme Poverty
             * 2 : Moderate Poverty
             * 3 : Vulnerable Households
             * 4 : Non-vulnerable Households"""
        st.subheader("Inference")
        target_inference
    if st.sidebar.checkbox("Check Family members information"):
        st.subheader("Check if all family members have the same poverty level or not")
        st.text(f'There are {check_all_member_same_target(income_train_df)} households where the family members do not have same poverty level') 
        st.subheader("Check if all families are with Heads")
        st.text(f'There are {check_with_head_or_not(income_train_df)} families without family Head')  
        st.subheader("Check if families without Heads have different poverty levels") 
    if st.sidebar.checkbox('Check Dataset information'):
        st.subheader("Dataset Sample")
        st.write(income_train_df.iloc[:50])
        st.subheader("Dataset Shape")
        st.text(f"The Training Dataset Consists of {income_train_df.shape[0]} rows and {income_train_df.shape[1]} columns")
        st.subheader("Columns Info")
        data_dim=st.radio("Column type information",("float_columns","int_columns","object_columns"))
        if data_dim=="float_columns":
            st.text(f'Out of 143 {len(float_columns)} are Float_type type columns')
            st.write(float_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[float_columns]))
            st.subheader("Inference")
            st.text(
                """                 *   v2a1 6860 ==>71.77%  variable explaination => Monthly rent payment
                *   v18q1 7342 ==>76.82% variable explaination => Number of tablets household owns
                *   rez_esc 7928 ==>82.95% variable explaination => Years behind in school
                *   meaneduc 5 ==>0.05% variable explaination => average years of education for adults (18+)
                *   SQBmeaned 5 ==>0.05% variable explaination => square of the mean years of education of adults""")
        if data_dim=="object_columns":
            st.text(f'Out of 143 {len(object_columns)} are Categorical type columns')
            st.write(object_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[object_columns]))
            st.subheader("Inference")
            st.text('There Are no null values in object columns')
        if data_dim=="int_columns":
            st.text(f'Out of 143 {len(int_columns)} are Integer type columns')
            st.write(int_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[int_columns]))
            st.subheader("Inference")
            st.text('There Are no null values in integer type columns')
        
        
        #st.write()
        
    
    
    
if __name__=='__main__': #check for main executed when programme is called 
    main()