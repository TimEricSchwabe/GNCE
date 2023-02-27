import random

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
import json
import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import os
import sys

# Adding edited version of pyrdf2vec to path
sys.path.append("/home/tim/pyRDF2Vec/pyrdf2vec")

def uri_to_id(uri):
    return uri.split('/')[-1]


"""
This file manages the generation of embeddings for the different entities

"""


uri_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{<{}> ?p ?o.}}
        UNION
        {{?s <{}> ?o.}}
        UNION
        {{?s ?p <{}>.}}

        }}
        LIMIT 30000000000
        """

literal_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{?s ?p '{}'.}}

        }}
        LIMIT 3000000000000
        """


def get_embeddings(dataset_name, kg_file, entities=None, remote=True, sparql_endpoint="http://127.0.0.1:8902/sparql/"):
    """
    This function calculates simple occurences as well as rdf2vec embeddings for a given rdf graph
    Saves the embeddings and occurrences in one json file per entity
    :param dataset_name: The name of the kg, used to save the statistics
    :param kg_file: path to the .ttl file of the kg
    :param entities: list of entities for which to calculate embeddings
    :param remote: flag whether a remote sparql endpoint will be used or the .ttl file in memory
    :param sparql_endpoint: url of the remote sparql endpoint, if used
    :return: None
    """
    #GRAPH = KG(kg_file, skip_verify=True)
    GRAPH = KG(sparql_endpoint, skip_verify=True)


    if remote:
        entities = entities
    else:
        if entities is not None:
            entities = entities
        else:
            train_entities = [entity.name for entity in list(GRAPH._entities)]
            test_entities = [entity.name for entity in list(GRAPH._vertices)]
            entities = set(train_entities + test_entities)

        #Create the RDF2vec model
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10, vector_size=100),
        walkers=[RandomWalker(4, 5, with_reverse=True, n_jobs=2, md5_bytes=None)],
        verbose=2, batch_mode='onefile'
    )


    # Count Occurences of nodes
    occurrences = {}


    elements_to_remove = []
    print("Calculating Occurences")
    occurrence_from_file = True

    if occurrence_from_file:
    #if False:
        file = open(kg_file, "r")
        Lines = file.readlines()

        for line in tqdm(Lines):
            line = line.split(" ")
            s = line[0].replace("<", "").replace(">", "")
            p = line[1].replace("<", "").replace(">", "")
            o = line[2].replace("<", "").replace(">", "")
            try:
                occurrences[s] += 1
            except KeyError:
                occurrences[s] = 1
            try:
                occurrences[p] += 1
            except:
                occurrences[p] = 1
            try:
                occurrences[o] += 1
            except:
                occurrences[o] = 1
        del Lines
        file.close()
    else:
        raise NotImplementedError

    #Saving Occurences
    with open(dataset_name + "_ocurrences.json", "w") as fp:
        json.dump(occurrences, fp)



    # Generate the embeddings
    print("Starting to fit model")
    #embeddings, literals = transformer.fit(GRAPH, entities)
    transformer.fit(GRAPH, entities)
    print("Finished fitting model")

    # Generating embedding for all entities
    test_entities_cleaned = []
    embeddings_test = []
    occurences_test = []
    i = -1
    print("Calculating Embeddings")
    for entity in tqdm(entities):
        i += 1
        try:
            embedding, literals = transformer.transform(GRAPH, [uri_to_id(entity)])
            test_entities_cleaned.append(entity)
            embeddings_test += embedding
            occurences_test.append(occurrences[entity])
        except:
            print(entity)
            raise


    print(len(occurrences))
    print(len(test_entities_cleaned))

    # Storing embeddings one by one to separate files(necessary for large KG)
    print("Saving statistics")
    for i in tqdm(range(len(test_entities_cleaned))):
        statistics_dict = {"embedding": embeddings_test[i].tolist(), "occurence": occurences_test[i]}

        file_name = test_entities_cleaned[i].replace("/", "|")
        with open(os.path.join("Datasets", dataset_name, "statistics", file_name + ".json"), "w") as fp:
            json.dump(statistics_dict, fp)
    return
if __name__ == "__main__":

    QUERY_FILE_PATH = '/home/tim/cardinality_estimator/Datasets/yago/star/Joined_Queries.json'
    KG_FILE_PATH = "Datasets/yago/graph/yago.ttl"
    KG_ENDPOINT = "http://localhost:8906/sparql/"
    KG_NAME = "yago"


    #Get entities from queries:
    entities = []
    # Joined Queries
    with open(QUERY_FILE_PATH, 'r') as f:
        queries = json.load(f)
    for query in queries:
        entities += query['x']

    entities = list(set(entities))
    print('Using ', len(entities), ' entities for RDF2Vec')


    print('Starting...')
    get_embeddings(KG_NAME, KG_FILE_PATH, remote=True, entities=entities, sparql_endpoint=KG_ENDPOINT)



