import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to read messages and categories data
    
    Input:
    messages_filepath - path to csv file with messages data
    categories_filepath - path to csv file with categories data
    
    Output:
    df - dataframe with merged messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """
    Function to clean dataframe. Spliting column 'categories' to the 
    single columns and concatenate it with df. Removing duplicates.
    
    Input:
    df - dataframe with merged messages and categories data
    
    Output:
    df - dataframe with cleaned data and splitted 'categories' column
    """
    #split category column and rename it
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = list(map(lambda x: x[:-2], categories.iloc[0]))
    categories.columns = category_colnames
    
    #replace values with last character
    for column in categories:
        categories[column] = categories[column].map(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
        
    # delete old category column
    if 'categories' in df: del df['categories']
        
    #join splited columns
    df = pd.concat([df, categories], axis=1)
    #drop duplicates
    df = df.drop_duplicates()
    print(df.head())
    return df


def save_data(df, database_filename):
    """
    Function to save dataset in SQLite Database.
    
    Input:
    df - cleaned dataframe
    database_filename - path where dataframe should be saved
    """
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    print(table_name)
    df.to_sql('datatable', engine, index=False, if_exists='replace')
    pass


def main():
    """
    Main function, which read, clean and save dataset.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()