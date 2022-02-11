import argparse
from re import I
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description="This script will generate possible descriptions for your Codenames game.")
    parser.add_argument("--input", "-i", required=True, help="Input text file that first row contains good words, and second row contains bad words. Words are separated by tab")
    # TODO: include neutral words
    parser.add_argument("--embedding", "-e", required=True, help="Word embedding used to find best result")
    parser.add_argument("--embedding_type", "-t", required=True, help="Word embedding type. Default is GLoVE")
    # TODO: add model/embedding selection: GLoVE, BERT, etc.

    return parser.parse_args()


def load_embeddings_and_words(input_path, embedding_path, type):
    if (type.lower() == "glove"):
        embeddings = []
        good_word_ids = []
        bad_word_ids = []
        words = []
        with open(embedding_path, "r") as glove, open(input_path, "r") as input_file:
            rows = input_file.readlines()
            good_words = set(rows[0].lower().split("\t"))
            bad_words = set(rows[1].lower().split("\t"))

            embeddings = []
            i=0
            for row in glove:
                word, embedding = row.split(maxsplit=1)
                word_lower = word.lower()
                embeddings.append(np.fromstring(embedding, "f", spe=" "))
                words.append(word_lower)

                if(word_lower in good_words):
                    good_word_ids.append(i)

                if(word_lower in bad_words):
                    bad_word_ids.append(i)

                i=i+1
            
            embeddings = np.array(embeddings)
            print("embedding shape: " + str(embeddings.shape))
            return good_word_ids, bad_word_ids, words, embeddings


def get_top_results(good_word_ids, bad_word_ids, words, embeddings, top=20):
    """
    Use matrix dot product rather than loop thru embeddings and scipy distance.
    This way should be more efficient.
    """

    good_word_embeddings = embeddings[good_word_ids, :]
    bad_word_embeddings = embeddings[bad_word_ids, :]

    # calculate cosine distance of all words and good/bad words.
    good_word_dot_product = np.sum(good_word_embeddings * embeddings, axis=1) # axis=1 sum across columns.
    print("good_word_dot_product size: " + str(good_word_dot_product.shape))
    bad_word_dot_product = np.sum(bad_word_embeddings * embeddings, axis=1)
    print("bad_word_dot_product size: " + str(bad_word_dot_product.shape))

    good_word_mod = np.sqrt(np.sum(good_word_embeddings * good_word_embeddings))
    bad_word_mod = np.sqrt(np.sum(bad_word_embeddings * bad_word_embeddings))
    mod_embedding = np.sqrt(np.sum(embeddings*embeddings, axis=1))

    good_word_cos_distance = 1- good_word_dot_product/(good_word_mod*mod_embedding) # sort by min-max dist?
    bad_word_cos_distance = 1- bad_word_dot_product/(bad_word_mod*mod_embedding)


    # calculate score (kinda like a loss function, but we maximize this)
    # score = log(1-good_word_distance) + log(bad_word_distance)
    score_matrix = math.log(1-good_word_cos_distance) + math.log(bad_word_cos_distance)
    score_matrix[::-1].sort() # sort in desc order
    top_candidates_ids = score_matrix[0:top]

    print(top_candidates_ids)

    top_words = []
    for id in top_candidates_ids:
        top_words.append(words[id])

    return top_words


if __name__ == "main":
    args = parse_args()
    good_word_ids, bad_word_ids, words, embeddings = load_embeddings_and_words(args.embedding, args.embedding_type)
    top_candidates = get_top_results(good_word_ids, bad_word_ids, words, embeddings)