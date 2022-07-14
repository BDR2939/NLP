import pandas as pd 
import numpy as np
import glob
import os 
from pathlib import Path
import random
import time

def merge_between_2_df(df1, df2):
    """
    Parameters
    ----------
    df1 : dataframe
    df2 : dataframe

    Returns
    -------
    merge_df : df
        merge of 2 dataframes base unique columns
    """
    columns1 = df1.columns.tolist()
    columns2 = df2.columns.tolist()
    
    diff_columns  = list(set(columns1)-set(columns2))
    uniuqe_columns   = list(set(columns1)-set(diff_columns))
    
    merge_df = pd.merge(df1, df2, how = 'inner', on = uniuqe_columns )
    return merge_df


def convert_location_to_list_of_ranges(train_df):
    """
    Parameters
    ----------
    train_df : dataframe

    Returns
    -------
    train_df : dataframe
        return train_df after ordering localization string into ranges of ranges.
        index to anotation index in pn notes
    """
    location_list  = train_df['location'].tolist()
    amount_of_locations = []
    for i_location_idx  in range(location_list.__len__()):
        if i_location_idx == 6:
            a=5
        # print(i_location_idx)
        i_location = location_list[i_location_idx]
        if i_location.find(',')!=-1:
            amount_of_locations.append(i_location.count(',')+1)
            i_location_without_quate  = i_location.split('[')[1].split(']')[0].replace("'", '')
            i_location_without_quates_splited  = i_location_without_quate.split(',')
            range_list = []
            for i_range_idx, i_range in enumerate(i_location_without_quates_splited):
                if i_range.find(';')==-1:
                    i_range_splited = i_range.split(' ')
                    if '' in i_range_splited:
                        i_range_splited.remove('')
                    
                    curr_range = list(map(lambda x: int(x), i_range_splited))
                    range_list.append(curr_range)
    
                else:
                    i_range_splited = i_range.split(';')
                    range_list2 = []
                    for i_range_idx2, i_range2 in enumerate(i_range_splited):
                        i_range_splited = i_range2.split(' ')
                        if '' in i_range_splited:
                            i_range_splited.remove('')
                        
                        curr_range = list(map(lambda x: int(x), i_range_splited))
                        range_list2.append(curr_range)
                    range_list.append(range_list2)
    
            location_list[i_location_idx] = range_list
    
        elif len(i_location) == 2:
            location_list[i_location_idx] = []
            amount_of_locations.append(0)
        else:
            amount_of_locations.append(1)
            i_location_without_quate  = i_location.split('[')[1].split(']')[0].replace("'", '')
    
            if i_location_without_quate.find(';')==-1:
                i_range_splited = i_location_without_quate.split(' ')
                curr_range = list(map(lambda x: int(x), i_range_splited))
                location_list[i_location_idx] = [curr_range]
            else:
                i_range_splited = i_range.split(';')
                range_list2 = []
                for i_range_idx2, i_range2 in enumerate(i_range_splited):
                    i_range_splited = i_range2.split(' ')
                    if '' in i_range_splited:
                        i_range_splited.remove('')
                    
                    curr_range = list(map(lambda x: int(x), i_range_splited))
                    range_list2.append(curr_range)
                location_list[i_location_idx] = [range_list2]
    
    train_df['location']  = location_list      
    train_df['location_splits']  = amount_of_locations
    return   train_df 



def get_annotation(merge_train_feature_df):
    """
    Parameters
    ----------
    merge_train_feature_df : dataframe

    Returns
    -------
    merge_train_feature_df : dataframe
        get from pn note the anotation.

    """
    location_list = merge_train_feature_df['location'].tolist()
    pn_history_list = merge_train_feature_df['pn_history'].tolist()
    
    all_localization_text = []
    for i_localization_idx, i_localization in enumerate(location_list):
        print(i_localization_idx)
        if i_localization.__len__()== 0:
            all_localization_text.append([])
        else:
            i_pn_history= pn_history_list[i_localization_idx]
            text = []
            for j_localization_idx, j_localization in enumerate(i_localization):
                if isinstance(j_localization[0], list):
                    text2 = []
                    for l_localization_idx, l_localization in enumerate(j_localization):
                        i_text = i_pn_history[l_localization[0]:l_localization[1]]
                        text2.append(i_text)
                    text.append(text2)
                else:
                    text.append(i_pn_history[j_localization[0]:j_localization[1]])
    
            all_localization_text.append(text)
    
        
        
    merge_train_feature_df['text_base_location'] = all_localization_text
    return merge_train_feature_df


def generate_train_batch_data(merge_train_feature_df, batch_size = 8):
    """
    Parameters
    ----------
    merge_train_feature_df : dataframe
        merge of 3  data sets - feaure\train\pn note.
    batch_size : int, optional
        batch size for the training. The default is 8.

    Returns
    -------
    input_list : list
        all input in list.
    tag_list : list
        all tags in list.
    input_tag_list : list
        all input and tags togther.
    """
    pn_num_list = merge_train_feature_df['pn_num'].tolist()
    pn_num_unique = np.unique(pn_num_list)
    input_list  = []
    tag_list = []
    input_tag_list  = []
    batch_input_list = []
    batch_tag_list = []
     
    for pn_num_idx, pn_num in enumerate(pn_num_unique):
        # print(pn_num_idx)
        if pn_num_idx == 8:
            a=5
        if pn_num_idx%batch_size ==0 and pn_num_idx !=0:
            batch_input_array = np.concatenate(batch_input_list)
            batch_tag_array = np.concatenate(batch_tag_list)
            input_list.append(batch_input_array)
            tag_list.append(batch_tag_array)
            input_tag_list.append((batch_input_array, batch_tag_array))
            batch_input_list = []
            batch_tag_list = []
        
        pn_tag = np.zeros((1,pn_num_unique.size))
        i_pn_df  = merge_train_feature_df.loc[merge_train_feature_df['pn_num'] == pn_num]
        i_pn_feature_num = i_pn_df['unique_index'].to_numpy()
        np.put(pn_tag, i_pn_feature_num, i_pn_feature_num.__len__()*[1])
        i_pn_input  = i_pn_df.iloc[0][ 'pn_history_splited']
        batch_input_list.append(i_pn_input)
        batch_tag_list.append(pn_tag)
        
        
    if  batch_tag_list.__len__() !=  batch_size:
        curr_batch_size =  batch_tag_list.__len__()
        amount_of_sample_2_generate = batch_size-curr_batch_size
        random_indexs = random.sample(range(0, (pn_num_unique.size- curr_batch_size)), amount_of_sample_2_generate)
        for i_idx in random_indexs:
            pn_tag = np.zeros((1,pn_num_unique.size))
            i_pn_df  = merge_train_feature_df.loc[merge_train_feature_df['pn_num'] == pn_num_unique[i_idx]]
            i_pn_feature_num = i_pn_df['unique_index'].to_numpy()
            np.put(pn_tag, i_pn_feature_num, i_pn_feature_num.__len__()*[1])
            i_pn_input  = i_pn_df.iloc[0][ 'pn_history_splited']
            batch_input_list.append(i_pn_input)
            batch_tag_list.append(pn_tag)
        batch_input_array = np.concatenate(batch_input_list)
        batch_tag_array = np.concatenate(batch_tag_list)
        input_list.append(batch_input_array)
        tag_list.append(batch_tag_array)
        input_tag_list.append((batch_input_array, batch_tag_array))
    
    return input_list, tag_list, input_tag_list
    

def merge_between_patient_note_train_and_feature(patient_notes_df, features_df, train_df):
    """
    Parameters
    ----------
    patient_notes_df : dataframe
        patient doctors notes.
    features_df : dataframe
        feature options in dataframe.
    train_df : dataframe
        data to train.

    Returns
    -------
    merge_train_feature_df : dataframe
        merge of 3  data sets - feaure\train\pn note.
    """
    
    # !todo - or\roni to sellect spicial caracter to remove 
    spicial_charchther_to_remove = [ ]
    for i_spicial_char in spicial_charchther_to_remove:
        patient_notes_df['pn_history_splited'] = patient_notes_df['pn_history'].apply(lambda x: x.replace('', i_spicial_char))
    patient_notes_df['pn_history_splited'] = patient_notes_df['pn_history'].apply(lambda x: x.lower().split(' '))
    features_df['feature_text'] = features_df['feature_text'].apply(lambda x: x.lower())
    unique_features = features_df.feature_text.unique()
    unique_feature_text_df = pd.DataFrame(unique_features, columns = ['feature_text'])
    unique_feature_text_df['unique_index'] =  np.arange(0, unique_feature_text_df.shape[0])
    features_df  = pd.merge(unique_feature_text_df, features_df, how = 'inner', on = ['feature_text'] )
    train_df['annotation'] = train_df['annotation'].apply(lambda x: x.lower())
    merge_train_feature_df = merge_between_2_df(train_df, features_df)
    merge_train_feature_df = merge_between_2_df(merge_train_feature_df, patient_notes_df)
    
    return merge_train_feature_df



def roni_generate_data(features_df, patient_notes_df,train_df ):
    """
    Parameters
    ----------
    features_df : TYPE
        DESCRIPTION.
    patient_notes_df : TYPE
        DESCRIPTION.
    train_df : TYPE
        DESCRIPTION.

    Returns
    -------
    patient_notes_train_data : TYPE
        DESCRIPTION.
    """
    unique_features = features_df.feature_text.unique()
    case_num = []
    feature_num = []
    unique_feature_num = []

    for i, feature in enumerate(unique_features):
        case_num = case_num + features_df.loc[features_df['feature_text'] == feature, 'case_num'].to_list()
        feature_num = feature_num + features_df.loc[features_df['feature_text'] == feature, 'feature_num'].to_list()
        unique_feature_num = unique_feature_num + list(np.full(len(case_num) - len(unique_feature_num), i))

    feature_mapping = pd.DataFrame({'case_num': case_num, 'feature_num': feature_num, 'unique_feature_num': unique_feature_num})


    train_df['unique_feature_num'] = train_df.apply(lambda raw: feature_mapping.loc[(feature_mapping['case_num'] == raw['case_num']) & (feature_mapping['feature_num'] == raw['feature_num']), ['unique_feature_num']], axis = 1)
    train_df.location.replace('[]', None, inplace=True)
    train_df.dropna(inplace = True, axis = 0)

    patient_notes = patient_notes_df
    patient_notes_train_data = patient_notes.loc[patient_notes['pn_num'].isin(train_df.pn_num), :].copy()
    patient_notes_train_data['features'] = patient_notes_train_data.apply(lambda raw: train_df.loc[train_df['pn_num'] == raw['pn_num'], 'unique_feature_num'].to_list(), axis = 1)
    return patient_notes_train_data

# -------------------------------------------------------------------------------------------------------------------------------------------

user = 'or '
if user == 'or':
    csv_folder_path = r'C:\MSC\NLP2\project\nbme-score-clinical-patient-notes'
else:
    csv_folder_path = r'C:\MSC\NLP2\project\nbme-score-clinical-patient-notes'

"""
there are 5 csv files:
 1. features
 2. patient_notes
 3. sample_submission
 4. test
 5. train
"""

# read  all csv from folder
csv_path_list = glob.glob(os.path.join(csv_folder_path, '*.csv'))
csv_dict = {}
for i_path in csv_path_list:
    file_name = Path(i_path).stem
    csv_df = pd.read_csv(i_path)
    csv_dict[file_name] = csv_df


# get csv's
patient_notes_df  =  csv_dict['patient_notes']
features_df  =  csv_dict['features']
train_df  =  csv_dict['train']

# can be remove not realy needed to order the localicalization
order_localization = False
if order_localization:
    train_df = convert_location_to_list_of_ranges(train_df)

# merge between 3 data frames 
merge_train_feature_df = merge_between_patient_note_train_and_feature(patient_notes_df, features_df, train_df)

# generate data for training
input_list, tag_list, input_tag_list = generate_train_batch_data(merge_train_feature_df, batch_size = 32)

# -------------------------------------------------------------------------------------------------------------------------------------------








