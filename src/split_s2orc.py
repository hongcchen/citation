import glob
from tqdm import tqdm
import json
import nltk
import os
import pandas as pd

from estimate_similarity import *

from multiprocessing import Process, Queue

def split_sentences(text):
    text = text.replace("et al.", "et al ")
    sentences = nltk.tokenize.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence) > 3]
    return sentences

def compute_embeddings(sentences, gpu_id):
# def compute_similarity_for_batch(single_strings, list_of_strings_batch, gpu_id):
    # Set the GPU to be used for this process
    torch.cuda.set_device(gpu_id)

    estimator = SimilarityEstimator(device=f"cuda:{gpu_id}")

    # Assuming the model is initialized here and moved to the specified GPU
    # model = Model().cuda(gpu_id)

    # Note: We assume that estimate_ims can handle batches efficiently.
    # Compute the similarity for the batch and do whatever is necessary

    # batch_similarity_lists = estimator.estimate_ims_list_of_list(single_strings, list_of_strings_batch)
    # return batch_similarity_lists
    # return_list.extend(batch_similarity_lists)

    embeddings = estimator.get_embeddings(sentences)

    # Convert the numpy array to a list
    list_embeddings = embeddings.tolist()

    return embeddings

def worker(input_queue, output_queue, gpu_id):
    # Set GPU
    torch.cuda.set_device(gpu_id)
    # Instantiate the model once per worker
    estimator = SimilarityEstimator(device=f"cuda:{gpu_id}")

    while True:
        task = input_queue.get()

        if isinstance(task, str) and task == "DONE":
            break

        corpusid, content = task
        sentences = split_sentences(content)
        embeddings = estimator.get_embeddings(sentences)
        list_embeddings = embeddings.tolist()

        formatted_data = {
            "corpusid": corpusid,
            "sentences": sentences,
            "embeddings": list_embeddings
        }

        output_queue.put(formatted_data)

        # Empty the GPU cache after processing each batch
        # torch.cuda.empty_cache()

def process_file(filename, base_name, gpus = [0, 1, 3, 4, 5, 6]):

    if not gpus:
        raise ValueError("No GPU IDs provided.")

    # Queues for task distribution and result collection
    input_queue = Queue()
    output_queue = Queue()

    # Initialize task counter
    num_tasks = 0

    # Start worker processes
    processes = [Process(target=worker, args=(input_queue, output_queue, gpus[i % len(gpus)])) for i in range(len(gpus))]
    for p in processes:
        p.start()

    with open(filename, 'r') as file_in_process:
        for line in file_in_process:
            line = line.strip()  # strip to remove newline characters
            if not line:  # skip empty lines
                continue
            try:
                data = json.loads(line)
                corpusid = data.get("corpusid")
                content = data.get("content")["text"]
                # Check if none of the values are None
                if all([corpusid, content]):
                    # Send tasks to workers
                    input_queue.put((corpusid, content))
                    num_tasks += 1  # Increment the task counter

                    # if num_tasks > 3:
                    #     break

            except json.JSONDecodeError:
                print(f"Error decoding JSON for line: {line}")
                continue

    # Signal workers to finish
    for _ in processes:
        input_queue.put('DONE')

    # Collect results
    BATCH_SIZE = 10_000  # or whatever size makes sense
    batch = []
    batch_count = 0

    # for _ in tqdm(range(num_batches), desc="Collecting results"):
    #     similarity_lists.extend(output_queue.get())
    # dfs = []
    for _ in tqdm(range(num_tasks), desc="Collecting results"):
        formatted_data = output_queue.get()
        batch.append(formatted_data)

        if len(batch) == BATCH_SIZE:
            df = pd.DataFrame(batch)
            df.to_parquet(os.path.join(OUTPUT_DIR, f"{base_name}_{batch_count}.parquet"))
            batch = []
            batch_count += 1

    # Wait for processes to finish
    for p in processes:
        p.join()

    # return dfs

# def callback_function(result):
#     output_data, filename = result
#     # Extract the base name (without extension) and directory
#     base_name = os.path.basename(filename).split('.')[0]
#
#     # Create a unique TSV filename based on the input filename
#     output_file = os.path.join(OUTPUT_DIR, f"{base_name}_processed.tsv")
#
#     df = pd.DataFrame(output_data)
#     df.to_csv(output_file, sep='\t', index=False)
#
#
# def error_callback(e):
#     print("Error:", e)

INPUT_DIR = '/shared/3/projects/citation-context/s2orc/s2orc/'
OUTPUT_DIR = '/shared/3/projects/citation-context/s2orc/s2orc_split/'
INPUT_FILE_LIST = [x for x in glob.glob(INPUT_DIR + "s2orc_*")]

if __name__ == "__main__":
    # num_processes = 1  # or min(26, len(INPUT_FILE_LIST))
    # pool = multiprocessing.Pool(processes=num_processes)
    # for file in INPUT_FILE_LIST:
    #     pool.apply_async(process_file, (file,), callback=callback_function, error_callback=error_callback)

    nltk.download('punkt')
    print(len(INPUT_FILE_LIST) + 1)
    for file in INPUT_FILE_LIST:
        print(file)
        base_name = os.path.basename(file).split('.')[0]
        # Create a unique TSV filename based on the input filename
        output_file_first = os.path.join(OUTPUT_DIR, f"{base_name}_0.parquet")

        if not os.path.exists(output_file_first):
            print(output_file_first)
            process_file(file, base_name, gpus=[0, 1, 3, 4, 5, 6])
            # formatted_data_list = process_file(file, gpus=[0, 1, 3, 4, 5, 6])
            # formatted_data_list = process_file(file, gpus=[0])
            # print(type(formatted_data_list[0]["embeddings"]))
            # print(len(formatted_data_list[0]["embeddings"]))
            # print(len(formatted_data_list[0]["embeddings"][0]))
            '''
            It is converted to list of arrays
            each array is of length 768
            so len(embendding[0]) = 768
            '''
            # df = pd.DataFrame(formatted_data_list)
            # print(df.dtypes)
            # df.to_parquet(output_file)
            # df.to_csv(output_file, index=False, sep='\t')

