import os
import numpy as np
import time

from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS

import pgvector.psycopg
import pgvector.asyncpg
import psycopg
from pprint import pprint
from dotenv import load_dotenv


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if PINECONE_API_KEY is None:
    print("PINECONE_API_KEY not found in .env file")
else:
    print("PINECONE_API_KEY loaded successfully. it is %s" % PINECONE_API_KEY)


socket_dir = "/home/ubuntu/pg_sockets"


class PgvectorIndex(BaseFilterANN):

    def __init__(self,  metric, index_params):
        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def notice_handler(self, notice):
        print("Received notice:", notice.message_primary)
    
    def fit(self, dataset):
        ds = DATASETS[dataset]()

        if ds.search_type() != "knn_filtered":
            raise NotImplementedError()

        print(f"Building index")
        # TODO: create the index here using the train dataset
        
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True, host=socket_dir)
        pgvector.psycopg.register_vector(conn)
        
        # send client messages to stdout
        conn.add_notice_handler(self.notice_handler)
        
        cur = conn.cursor()
        cur.execute("SET client_min_messages = 'NOTICE'")
        cur.execute("SET pinecone.top_k = 100")
        self._cur = cur
        
        skip_index_build = False
        
        if skip_index_build:
            return
        
        X = ds.get_dataset()
        filter_metadata = ds.get_dataset_metadata()
        non_zero_column_indices = np.nonzero(filter_metadata)
        
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d), filter_1 int, filter_2 int)" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        start = time.time()
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                # check if the embedding is all zeros and if so skip and warn
                if np.all(embedding == 0):
                    print(f"embedding {i} is all zeros!!!")
                    continue
                if not i % 1000:
                    print(i)
                    print(time.time() - start, "seconds")
                if i > 20000:
                    pass # no effect
                filter_tags = non_zero_column_indices[1][non_zero_column_indices[1] == i]
                copy.write_row((i, embedding, filter_tags[0], filter_tags[1]))
        print("done copying data")
        return
    
            

    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.

        Checking the index usually involves the dataset name
        and the index build parameters passed during construction.

        If the file does not exist, there is an option to download it from a public url
        """
        # the code ideally expects to load the index in this variable but in our design the index will be inside postgres db
        self.index = None
        return True


    def index_files_to_store(self, dataset):
        """
        Specify a triplet with the local directory path of index files,
        the common prefix name of index component(s) and a list of
        index components that need to be uploaded to (after build)
        or downloaded from (for search) cloud storage.

        For local directory path under docker environment, please use
        a directory under
        data/indices/track(T1 or T2)/algo.__str__()/DATASETS[dataset]().short_name()
        """
        raise NotImplementedError()
    
    def query(self, X, k):
        raise NotImplementedError()

    def filtered_query(self, X, filter, k):

        # if (X.dtype.kind == 'f'):
        #     print('data type of X is ' + str(X.dtype))
        #     X = X*10 + 128
        #     X = X.astype(np.uint8)
        #     padding_size = 192 - X.shape[1]
        #     X = np.pad(X, ((0, 0), (0, padding_size)), mode='constant')


        # results_tuple = self.index.search_parallel(X, filter.indptr, filter.indices, k) # this returns a tuple: (results_array, query_time, post_processing_time)
        # self.I = results_tuple[0]
        # print("query and postprocessing times: ", results_tuple[1:])


    def get_results(self):
        return self.I

    def set_query_arguments(self, query_args):
        self.qas = query_args


    def __str__(self):
        return f'pgvector_filter({self.indexkey, self._index_params, self.qas})'

   
