import pandas as pd
import numpy as np
from tqdm import tqdm
from estimate_similarity import *
import glob
import os
import ast

from multiprocessing import Process, Manager, Queue

def compute_similarity_for_batch(single_strings, list_of_strings_batch, gpu_id, return_list):
# def compute_similarity_for_batch(single_strings, list_of_strings_batch, gpu_id):
    # Set the GPU to be used for this process
    torch.cuda.set_device(gpu_id)

    estimator = SimilarityEstimator(device=f"cuda:{gpu_id}")

    # Assuming the model is initialized here and moved to the specified GPU
    # model = Model().cuda(gpu_id)

    # Note: We assume that estimate_ims can handle batches efficiently.
    # Compute the similarity for the batch and do whatever is necessary

    batch_similarity_lists = estimator.estimate_ims_list_of_list(single_strings, list_of_strings_batch)
    # return batch_similarity_lists
    return_list.extend(batch_similarity_lists)


def worker(input_queue, output_queue, gpu_id):
    # Set GPU
    torch.cuda.set_device(gpu_id)
    # Instantiate the model once per worker
    estimator = SimilarityEstimator(device=f"cuda:{gpu_id}")

    while True:
        task = input_queue.get()

        if isinstance(task, str) and task == "DONE":
            break

        # single_strings, list_of_strings_batch = task
        task = task[task["sentences"].apply(len) > 5]
        task['sentences'] = task['sentences'].apply(ast.literal_eval)
        single_strings = task['context'].values
        list_of_strings_batch = task['sentences'].values

        # batch_similarity_lists = estimator.estimate_ims_list_of_list(single_strings, list_of_strings_batch)
        # # Place results in the output queue
        # output_queue.put(batch_similarity_lists)

        task['similarity_list'] = estimator.estimate_ims_list_of_list(single_strings, list_of_strings_batch)
        output_queue.put(task)

        # Empty the GPU cache after processing each batch
        # torch.cuda.empty_cache()
def compute_similarity_for_dataframe_multiprocessing(df,output_file, batch_size=250, gpus=[1, 2, 3, 4, 5, 6]):
    """
    Compute similarities using multiple GPUs by processing data in batches.

    :param df: DataFrame containing data
    :param batch_size: Size of each batch
    :param gpus: List of GPU IDs to use
    :return: List of computed similarities
    """

    if not gpus:
        raise ValueError("No GPU IDs provided.")

    # Manager list to store results from all processes
    # manager = Manager()
    # similarity_lists = manager.list()

    # Determine the number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

    # Split the DataFrame into batches
    # single_string_batches = np.array_split(df['context'].values, num_batches)
    # list_of_strings_batches = np.array_split(df['sentences'].values, num_batches)

    df_batches = np.array_split(df, num_batches)

    # Queues for task distribution and result collection
    input_queue = Queue()
    output_queue = Queue()

    # Start worker processes
    processes = [Process(target=worker, args=(input_queue, output_queue, gpus[i % len(gpus)])) for i in range(len(gpus))]
    for p in processes:
        p.start()

    # Send tasks to workers
    # for single_strings, list_of_strings in zip(single_string_batches, list_of_strings_batches):
    #     input_queue.put((single_strings, list_of_strings))
    for df_batch in df_batches:
        input_queue.put(df_batch)

    # Signal workers to finish
    for _ in processes:
        input_queue.put('DONE')

    # Collect results
    # for _ in tqdm(range(num_batches), desc="Collecting results"):
    #     similarity_lists.extend(output_queue.get())
    dfs = []
    for _ in tqdm(range(num_batches), desc="Collecting results"):
        result = output_queue.get()
        dfs.append(result)
    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat.to_csv(output_file, index=False, sep='\t')
    # print(df_concat.iloc[0]['context'])
    # print(len(df_concat.iloc[0]['sentences']))
    # print(len(df_concat.iloc[0]['similarity_list']))


    # Wait for processes to finish
    for p in processes:
        p.join()

    # return list(similarity_lists)


def compute_similarity_for_dataframe(df, batch_size=250):
    similarity_lists = []

    # Determine the number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

    # Split the DataFrame into batches
    single_string_batches = np.array_split(df['context'].values, num_batches)
    list_of_strings_batches = np.array_split(df['sentences'].values, num_batches)

    for single_strings, list_of_strings in tqdm(zip(single_string_batches, list_of_strings_batches), total=num_batches,
                                                desc="Processing batches"):
        # Assuming the function compute_similarity_for_batch is defined somewhere
        batch_similarity_lists = compute_similarity_for_batch(single_strings, list_of_strings)
        similarity_lists.extend(batch_similarity_lists)

    return similarity_lists




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
            # df_s2orc = pd.read_csv(file, sep='\t')
            df_s2orc = pd.read_csv(file, sep='\t', nrows=1000)
            print(output_file)
            # df_s2orc = df_s2orc[df_s2orc["sentences"].apply(len) > 5]
            # print(df_s2orc.head(2))
            # df_s2orc['sentences'] = df_s2orc['sentences'].apply(ast.literal_eval)

            # df_s2orc['similarity_list'] = compute_similarity_for_dataframe(df_s2orc, batch_size=50)
            # df_s2orc['similarity_list'] = compute_similarity_for_dataframe_multiprocessing(df_s2orc, batch_size=1000, gpus=[0,1,3,4,5,6])
            # df_s2orc['similarity_list'] = compute_similarity_for_dataframe_multiprocessing(df_s2orc, batch_size=10, gpus=[0])
            # compute_similarity_for_dataframe_multiprocessing(df_s2orc,output_file=output_file, batch_size=1000, gpus=[0,1,3,4,5,6])

            compute_similarity_for_dataframe_multiprocessing(df_s2orc,output_file=output_file, batch_size=1000, gpus=[0,1,2])
            # print(df_s2orc.iloc[0]['context'])
            # print(len(df_s2orc.iloc[0]['sentences']))
            # print(len(df_s2orc.iloc[0]['similarity_list']))
            # os.rename(file, output_file)
            # df_s2orc.to_csv(output_file, index=False, sep='\t')
