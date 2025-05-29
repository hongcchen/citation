import pandas as pd
import numpy as np
from tqdm import tqdm
from estimate_similarity import *
import glob
import os
import ast

def compute_similarity_for_dataframe_multiprocessing(df,output_file, batch_size=250, gpu_id=0):
    """
    Compute similarities using multiple GPUs by processing data in batches.

    :param df: DataFrame containing data
    :param batch_size: Size of each batch
    :param gpu_id: GPU IDs to use
    :return: List of computed similarities
    """

    torch.cuda.set_device(gpu_id)
    estimator = SimilarityEstimator(device=f"cuda:{gpu_id}")

    # Determine the number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

    # Split the DataFrame into batches
    # single_string_batches = np.array_split(df['context'].values, num_batches)
    # list_of_strings_batches = np.array_split(df['sentences'].values, num_batches)

    df_batches = np.array_split(df, num_batches)

    dfs = []
    for task in tqdm(df_batches, desc="Collecting results"):
        # single_strings, list_of_strings_batch = task
        task = task[task["sentences"].apply(len) > 5]
        task['sentences'] = task['sentences'].apply(ast.literal_eval)
        single_strings = task['context'].values
        list_of_strings_batch = task['sentences'].values

        task['similarity_list'] = estimator.estimate_ims_list_of_list(single_strings, list_of_strings_batch)

        task['max_similarity'] = task['similarity_list'].apply(lambda x: max(x))
        task['max_index'] = task['similarity_list'].apply(lambda x: x.index(max(x)))
        task['the_sentence'] = task.apply(lambda x: x['sentences'][x['max_index']], axis=1)

        dfs.append(task)

    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat.to_csv(output_file, index=False, sep='\t')

INPUT_DIR = '/shared/3/projects/citation-context/s2orc/s2orc_merge/'
OUTPUT_DIR = '/shared/3/projects/citation-context/s2orc/s2orc_merge/'

# INPUT_FILE_LIST = glob.glob(os.path.join(INPUT_DIR, "s2orc_*_merged.tsv"))
INPUT_FILE_LIST = glob.glob(os.path.join(INPUT_DIR, "s2orc_*_merged_01.tsv"))

if __name__ == '__main__':

    for file in INPUT_FILE_LIST:
        base_name = os.path.basename(file).split('.')[0]
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_ims_added.tsv")

        if not os.path.exists(output_file):
        # if True:
            # Load the data
            # df_s2orc = pd.read_csv(file, sep='\t', nrows=10)
            # df_s2orc = pd.read_csv(file)
            print(file)
            df_s2orc = pd.read_csv(file, sep='\t')
            # df_s2orc = pd.read_csv(file, sep='\t', nrows=100)
            print(output_file)
            # df_s2orc = df_s2orc[df_s2orc["sentences"].apply(len) > 5]
            # print(df_s2orc.head(2))
            # df_s2orc['sentences'] = df_s2orc['sentences'].apply(ast.literal_eval)

            # df_s2orc['similarity_list'] = compute_similarity_for_dataframe(df_s2orc, batch_size=50)
            # df_s2orc['similarity_list'] = compute_similarity_for_dataframe_multiprocessing(df_s2orc, batch_size=1000, gpus=[0,1,3,4,5,6])
            # df_s2orc['similarity_list'] = compute_similarity_for_dataframe_multiprocessing(df_s2orc, batch_size=10, gpus=[0])
            # compute_similarity_for_dataframe_multiprocessing(df_s2orc,output_file=output_file, batch_size=1000, gpus=[0,1,3,4,5,6])

            compute_similarity_for_dataframe_multiprocessing(df_s2orc,output_file=output_file, batch_size=250, gpu_id=1)
            # print(df_s2orc.iloc[0]['context'])
            # print(len(df_s2orc.iloc[0]['sentences']))
            # print(len(df_s2orc.iloc[0]['similarity_list']))
            # os.rename(file, output_file)
            # df_s2orc.to_csv(output_file, index=False, sep='\t')
