import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe

    Input:
    messages_filepath   filepath to messages csv file
    categories_filepath filespath to categories csv file

    Returns:
    df  dataframe merging categories and messages
    '''
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories,on='id')
    
    return df

   


def clean_data(df):
    '''
    clean_data
    Take multiple preprocess steps to clean and preprocess the dataframe:
        1. Split the "categories" column into multiple columns , one for each category class
        2. Rename the class labels
        3. Convert category values to just numbers 0 or 1
        4. Drop duplicate rows

    Input:
    df  dataframe

    Returns:
    df cleaned dataframe 
    '''
    # Split categories into separate category columns.
    categories = df.categories.str.split(';',expand=True)
   
    # rename the column names of categories
    row = categories.iloc[0,:]
    category_colnames = row.str.split("-").str[0]
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
   
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)    

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # get rid of samples containing "2" in class label
    df = df.loc[~(df.loc[:,category_colnames]==2).any(axis=1),:]
    
    return df


def save_data(df, database_filename):
    '''
    save_data
    Store pandas dataframe in a SQLite database

    Input:
    df  dataframe
    database_file   database file address
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')
  


def main():
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