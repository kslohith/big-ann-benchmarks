import os
import numpy as np
import time
import sys
import random
import string

from tqdm import tqdm
from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
import pgvector.psycopg
import pgvector.asyncpg
import psycopg
from pprint import pprint
from dotenv import load_dotenv

# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Go up two levels to the parent of neurips23
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# print(parent_dir, current_dir)
# # Add the parent directory to sys.path
# sys.path.append(parent_dir)


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if PINECONE_API_KEY is None:
    print("PINECONE_API_KEY not found in .env file")
else:
    print("PINECONE_API_KEY loaded successfully. it is %s" % PINECONE_API_KEY)

socket_dir = "/var/run/postgresql"

class PgvectorRemoteIndex(BaseFilterANN):

    def __init__(self,  metric, index_params):
        self._metric = metric
        # self._m = index_params['M']
        # self._ef_construction = index_params['efConstruction']
        self._cur = None
        if metric == "angular":
            self._query = "SELECT id FROM ann_vectors where tags @> ARRAY[%s]::TEXT[] ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM ann_vectors where tags @> ARRAY[%s]::TEXT[] ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
        print("Instantiated pgvector-remote")
        
        # hack to make this code call the fit method. Need to figure out why runner is not calling fit on its own.
        self.fit("yfcc-10M")
        
        
    def notice_handler(self, notice):
        print("Received notice:", notice.message_primary)
        
    def _init_connection(self):
        if self._cur == None:
            conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True, host=socket_dir)
            pgvector.psycopg.register_vector(conn)
            
            # send client messages to stdout
            conn.add_notice_handler(self.notice_handler)
            
            cur = conn.cursor()
            # cur.execute("SET client_min_messages = 'DEBUG1'")
            cur.execute("SET pinecone.vectors_per_request = 100")
            cur.execute("SET pinecone.requests_per_batch = 40")
            
            self._cur = cur
    
    def fit(self, dataset):
        ds = DATASETS[dataset]()
        if ds.search_type() != "knn_filtered":
            raise NotImplementedError()

        print(f"Building index")
        self._init_connection()
        
        # set this flag to skip index building
        skip_index_build = False
        
        if skip_index_build:
            print('skipped index building')
            return
        
        max_insertion_limit = -1
        X = ds.get_dataset()[0:max_insertion_limit]
        filter_metadata = ds.get_dataset_metadata()[0:max_insertion_limit]
        non_zero_column_indices = np.nonzero(filter_metadata)
        
        print("drop table before")
        self._cur.execute("DROP TABLE IF EXISTS ann_vectors")
        print("drop table after")
        self._cur.execute(f'CREATE TABLE ann_vectors (id int, embedding vector({X.shape[1]}), tags TEXT[])')
        self._cur.execute("ALTER TABLE ann_vectors ALTER COLUMN embedding SET STORAGE PLAIN")
        
        print("copying data...")
        start = time.time()
        with self._cur.copy(f'COPY ann_vectors (id, embedding, tags) FROM STDIN') as copy:
            for i in tqdm(range(X.shape[0])):
                embedding = X[i]
                # print(i)
                # check if the embedding is all zeros and if so skip and warn
                if np.all(embedding == 0):
                    print(f"embedding {i} is all zeros!!!")
                    continue
                if not i % 10000:
                    print(i, time.time() - start, "seconds")
                # filter_tags = non_zero_column_indices[1][non_zero_column_indices[0] == i]
                filter_tags = filter_metadata[i].nonzero()[1]
                formatted_tags = "{" + ",".join(map(str, filter_tags)) + "}"
                # print(formatted_tags)
                copy.write_row((i, embedding, formatted_tags))
        print("done copying data")
        
        index_name =  'benchmark' + ''.join(random.choices(string.ascii_lowercase, k=4))
        print(f"...creating remote index: {index_name}")
        
        self._cur.execute("ALTER SYSTEM SET pinecone.api_key = '%s'" % PINECONE_API_KEY)
        self._cur.execute("SHOW pinecone.api_key")
        print(f'pinecone.api_key val in db: {self._cur.fetchone()}')
        # self._cur.execute(f'CREATE INDEX {index_name} ON ann_vectors USING pinecone (embedding, tags) with (spec = \'{"serverless":{"cloud":"aws","region":"us-west-2"}}\')')
        self._cur.execute(f'CREATE INDEX {index_name} ON ann_vectors USING pinecone (embedding, tags) with (spec = \'{{"serverless":{{"cloud":"aws","region":"us-west-2"}}}}\')')

        print(f"...done creating index in time: {time.time() - start} seconds")    
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
        print(f'Starting benchmark fetch queries. X.shape: {X.shape}, filter.shape: {filter.shape}')
        self._init_connection()
        self._cur.execute(f"SET pinecone.top_k = {k}")
        
        # X = X[0:2]
        # filter = filter[0:2]
        
        def get_tags(tag_data):
            non_zero_column_indices = np.nonzero(tag_data)
            ret = []
            n = tag_data.shape[0]
            for idx in range(n):
                tag_indices = non_zero_column_indices[1][non_zero_column_indices[0] == idx]
                # print(type(tag_indices.tolist()))
                ret.append(tag_indices.tolist())
            return ret
        
        start = time.time()
        tag_data = get_tags(filter)
        print(f"Filter construction took {time.time() - start} seconds")
                                
        result_list = []
        start = time.time()
        for i in range(X.shape[0]):
            self._cur.execute(self._query, (tag_data[i], str(X[i].tolist()), k), binary=True, prepare=True)
            result = [id for id, in self._cur.fetchall()]
            result_list.append(result)
            print(f'num: {i} tag: {tag_data[i]} result = {result}')
        print(f"Query took {time.time() - start} seconds")
        
        # print(result_list)
        
        
        max_length = k
        # Pad shorter sublists with zeros to make them uniform in length
        padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in result_list]
        # Convert the padded list to a NumPy array
        numpy_array = np.array(padded_list)

        self.I = np.array(numpy_array)
        # print(self.I)


    def get_results(self):
        return self.I

    def set_query_arguments(self, ef_search=100):
        self._init_connection()
        self._ef_search = ef_search
        # self._cur.execute("SET hnsw.ef_search = %d" % self._ef_search)


    def __str__(self):
        # return f'pgvector_filter({self.indexkey, self._index_params, self.qas})'
        return f'pgvector_filter'

   

# pg_vec = PgvectorIndex("euclidean", None)
# print("start")
# pg_vec.fit("yfcc-10M")
