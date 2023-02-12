import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

"""This File contains the reposiitory of all the function being used in the Income qualification project""" 

def find_target(df1,df2):
    """Will be used to find the target variable from the data sets 
    #function takes two datasets as input and provides with the column name that is not present in the test set""" 
    return (list(set(df1.columns)-set(df2.columns)))[0]
def analyse_catagorical_col(df,col_name,figsize):
    """Function can be used to analyse the categorical features would provide a countplot as an output for the provided (datafarame and column name and a tuple for figuresize) are the expected input variables  
    to adjust figsize""" 
    plt.figure(figsize=figsize)
    """forms a countplot for each category"""
    sns.countplot(x=df[f'{col_name}'])
    """Set the labels and title of the plot"""
    plt.title(f'{col_name} cardinality distribution')
    """Show the plot""" 
    plt.show()
def heat_map_coor_plot(var1,figsize):
    """This function creates a heatmap on the highly correlated features 
    takes a correlated matrix as one feature and figsize as s tuple"""
    plt.figure(figsize=figsize)
    sns.heatmap(var1,annot=True, cmap = sns.color_palette("Set2", 8), fmt='.3f')
def plot_null_values(df):
    """Function expects two inputs 1) the data frame 2) a tuple (width,height)
    calculating null value percentage column wise and allocating to a dataframe"""
    null_values = df.isnull().sum()/len(df)*100
    null_values = pd.DataFrame(null_values)
    """Reset the index to form it in proper dataframe"""
    null_values.reset_index(inplace=True)
    """Renaming columns""" 
    null_values.columns=['Feature','Percentage of missing values']
    return null_values
    
def replace_yes_no(df): 
    """This function replaces the Yes:1 and No:0 for the following columns and returns 
       the dataframe expects the the dataframe as input"""
    mapping = {'yes' :1, 'no' :0}
    for i in df:
        df['dependency']= df['dependency'].replace(mapping).astype(float)
        df['edjefe']= df['edjefe'].replace(mapping).astype(float)
        df['edjefa']= df['edjefa'].replace(mapping).astype(float)
        df['meaneduc']= df['meaneduc'].replace(mapping).astype(float)
    return df
def rep_null_val(df):
    """Replaces all null values the dataframe 
       for following column replacing with 0 as per the finsdings during discovery""" 
    df['v2a1'].fillna(0,inplace=True)
    """For following column replacing with 0 as per the findings during discovery"""
    df['v18q1'].fillna(0,inplace=True)
    """For following column replacing with 0 as per the findings during discovery"""
    df['rez_esc'].fillna(0,inplace=True)
    """For following column replacing with 'edjefe' columns respective value as per the findings during discovery"""
    df['meaneduc'].fillna(df['edjefe'],inplace=True)
    """For following column replacing with square of 'meaneduc' columns respective value as per the findings during discovery"""
    df['SQBmeaned'].fillna(df['meaneduc']**2,inplace=True)
    return df
def drop_columns(df):
    """This function takes the complete dataframe as input drops the unwanted columns and provides the exactly required dataframe""" 
    df.drop(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq','Id','idhogar','coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total', 'r4t3','area2'],axis=1,inplace=True)
    return df 
def cleaning_pipeline(df):
    """Function takes the dataframe then 
        1) replaces yes with 1 and no with 0 
        2) replaces all the null values in the reuired pattern as per discovery
        3) deletes all the unwanted columns"""  
    df1=replace_yes_no(df)
    df2=rep_null_val(df1)
    df3=drop_columns(df2)
    return df3