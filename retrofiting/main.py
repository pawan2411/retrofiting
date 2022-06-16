import getopt
import sys
import gensim
from gensim.models.keyedvectors import KeyedVectors
import copy

ITERATIONS = 10
"""
Name : main.py
Author : Pawan Kumar Rajpoot
Contect : pawan.rajpoot2411@gmail.com
Time    : 15th August 2021
Desc    : Solution for the question 1 for AI Lab | Verse Innovations Pvt. Ltd. Assignment
Reference   : https://www.cs.cmu.edu/~hovy/papers/15HLT-retrofitting-word-vectors.pdf
"""

def load_w2v(filename):
    '''
    :method: loads the vector file using gensim word2vec loader
    :param filename: file should be in glove format in either bin or txt extension
    :return: loaded dictionary
    '''
    if filename.endswith('.bin'):
        model = gensim.models.word2vec.Word2Vec.load_word2vec_format(filename, binary=True, no_header=False)
        gensim
    else:
        model = KeyedVectors.load_word2vec_format(filename, binary=False, no_header=False)
    return model


def load_ngraph(filename):
    '''
    :method: loads neighbourhood graph from the txt format file
    :param filename:
    :return: loaded dictionary with key as word and value as list of all connected words
    '''
    graph_dic = {}
    ngraph_file = open(filename, 'r')
    for sline in ngraph_file.readlines():
        all_nodes = sline.lower().strip().split()
        if len(all_nodes) > 1:
            graph_dic[all_nodes[0]] = all_nodes[1:]
        else:
            graph_dic[all_nodes[0]] = []
    return graph_dic


def retrofit_algo(old_embedding, neighbourhood_graph, iterations):
    '''
    :param old_embedding: old_embedding dictionary
    :param neighbourhood_graph: neighbourhood graph dictionary
    :param iterations: number of iterations (set to 10)
    :return: new dictionary of embeddings
    '''
    new_embedding = copy.deepcopy(old_embedding)
    common_nodes = set(old_embedding.index_to_key).intersection(set(neighbourhood_graph.keys()))
    for it in range(iterations):
        for word in common_nodes:
            neighbours_with_embeddings = set(neighbourhood_graph[word]).intersection(set(old_embedding.index_to_key))
            number_of_neighbours_with_embeddings = len(neighbours_with_embeddings)
            if number_of_neighbours_with_embeddings == 0:
                continue
            new_vector = number_of_neighbours_with_embeddings * old_embedding[word]
            for common_word_with_embedding in neighbours_with_embeddings:
                new_vector += new_embedding[common_word_with_embedding]
            new_embedding[word] = new_vector / (2 * number_of_neighbours_with_embeddings)
    return new_embedding


def write_new_embedding(embedding_dic, filename):
    '''
    :param embedding_dic: new evaluated embedding dictionary
    :param filename: where to save the new dictionary
    :return: None
    '''
    new_embed_file = open(filename, 'w', encoding='utf-8')
    count = 0
    for key in embedding_dic.index_to_key:
        value = embedding_dic[key]
        if count == 0:
            new_embed_file.write(str(len(embedding_dic.index_to_key)) + " " + str(len(value)) + "\n")
        value_str = ' '.join([str(num) for num in value])
        new_embed_file.write(key + " " + value_str + "\n")
        count = count + 1
    new_embed_file.close()


def read_input(argv):
    '''
    :method: parse the input arguments
    :param argv: system args
    :return: input embedding filename, output embedding filename, input neighbourhood graph filename
    '''
    inputfile = ''
    outputfile = ''
    ingraphfile = ''
    opts, args = getopt.getopt(argv, "hi:o:g:", ["ifile=", "ofile=", "gfile="])

    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfile> -o <outputfile> -g <graphfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg.strip()
        elif opt in ("-o", "--ofile"):
            outputfile = arg.strip()
        elif opt in ("-g", "--gfile"):
            ingraphfile = arg.strip()
    print(inputfile, outputfile, ingraphfile)
    return inputfile, outputfile, ingraphfile


if __name__ == '__main__':
    in_vector_file, out_vector_file, graph_file = read_input(sys.argv[1:])
    pretrained_embedding = load_w2v(in_vector_file)
    neighbourhood_graph = load_ngraph(graph_file)
    new_embeddings = retrofit_algo(pretrained_embedding, neighbourhood_graph, ITERATIONS)
    write_new_embedding(new_embeddings, out_vector_file)
