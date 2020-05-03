#!/usr/bin/env python3

import pandas as pd

from complexity_measures import dependency_based
from complexity_measures import constituent_based
from complexity_measures import utils


def main():
    metadata = pd.read_csv("/ccl/projects/Kallimachos/low_high_brow_corpus_v2/metadata.tsv", sep="\t", header=0, index_col=0)
    dep_based = [dependency_based.average_average_dependency_distance,
                 dependency_based.average_closeness_centrality,
                 dependency_based.average_outdegree_centralization,
                 dependency_based.average_closeness_centralization,
                 dependency_based.average_sentence_length,
                 dependency_based.average_dependents_per_word,
                 dependency_based.average_longest_shortest_path,
                 dependency_based.average_punctuation_per_sentence]
    const_based = [constituent_based.average_t_units,
                   constituent_based.average_complex_t_units,
                   constituent_based.average_clauses,
                   constituent_based.average_dependent_clauses,
                   constituent_based.average_nps,
                   constituent_based.average_vps,
                   constituent_based.average_pps,
                   constituent_based.average_coordinate_phrases,
                   constituent_based.average_constituents,
                   constituent_based.average_constituents_wo_leaves,
                   constituent_based.average_height]
    header = "id filename genre dependency_distance dependency_distance_stdev closeness_centrality closeness_centrality_stdev outdegree_centralization outdegree_centralization_stdev closeness_centralization closeness_centralization_stdev average_sentence_length average_sentence_length_stdev dependents_per_word dependents_per_word_stdev longest_shortest_path longest_shortest_path_stdev punctuation_per_sentence punctuation_per_sentence_stdev t_units t_units_stdev t_units_length t_units_length_stdev complex_t_units complex_t_units_stdev complex_t_units_length complex_t_units_length_stdev clauses clauses_stdev clauses_length clauses_length_stdev dependent_clauses dependent_clauses_stdev dependent_clauses_length dependent_clauses_length_stdev nps nps_stdev nps_length nps_length_stdev vps vps_stdev vps_length vps_length_stdev pps pps_stdev pps_length pps_length_stdev coordinate_phrases coordinate_phrases_stdev coordinate_phrases_length coordinate_phrases_length_stdev constituents constituents_stdev constituents_wo_leaves constituents_wo_leaves_stdev height height_stdev".split()
    print("\t".join(header))
    for idx, text in metadata.iterrows():
        with open("/ccl/projects/Kallimachos/low_high_brow_corpus_v2/%s" % text["filename"]) as f:
            dep_trees, const_trees = zip(*utils.read_txt_csv_graphs(f))
        print(idx, "\t".join(text), sep="\t", end="")
        for db in dep_based:
            score, stdev = db(dep_trees)
            print("", score, stdev, sep="\t", end="")
        for cb in const_based:
            result = cb(const_trees)
            if isinstance(result, tuple):
                result = "\t".join(str(r) for r in result)
            print("", result, sep="\t", end="")
        print()


if __name__ == "__main__":
    main()
