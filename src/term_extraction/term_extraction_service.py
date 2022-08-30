from collections import Counter, defaultdict
import math
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple
import re

from spacy.tokens import doc
import spacy.tokenizer
import spacy.language
from nltk.util import ngrams

TokenSequenceFilter = Callable[[List[Any]], List[List[str]]]


class Cvalue:
    """A class to compute the C-value of each term (token sequence) in a corpus of texts.
    """

    def __init__(self, corpus: Iterator[str], nlp: spacy.language.Language, tokenSeqFilter: TokenSequenceFilter, max_size_gram: int) -> None:
        self.corpus = corpus
        self.nlp = nlp
        self.tokenSeqFilter = tokenSeqFilter
        self.max_size_gram = max_size_gram

    def _filter_token_sequences(self) -> List[List[str]]:
        return self.tokenSeqFilter(self.nlp.pipe(self.corpus))

    def _count_token_sequences(self) -> Counter:
        return Counter([" ".join(tokens for tokens in self._filter_token_sequences())])

    def _order_candidate_terms(self, candidate_terms_by_size: Dict[str, int]) -> Tuple[List[str], Counter]:

        all_candidate_terms = []

        for terms_size, terms in candidate_terms_by_size.items():
            all_candidate_terms.extend(terms)
            candidate_terms_by_size[terms_size] = list(set(terms))

        candidateTermsCounter = Counter(all_candidate_terms)

        # each group of ngram needs to be ordered by the frequence
        for terms_size in candidate_terms_by_size.keys():
            candidate_terms_by_size[terms_size].sort(
                key=lambda term: candidateTermsCounter[term], reverse=True)

        candidateTerms = []
        for terms_size in range(1, self.max_size_gram + 1).__reversed__():
            candidateTerms.extend(candidate_terms_by_size[terms_size])

        return candidateTerms, candidateTermsCounter

    def _extract_candidate_terms(self) -> None:

        tokenSeqCounter = self._count_token_sequences()
        tokenSeqStrings = tokenSeqCounter.values()

        candidate_terms_by_size = defaultdict(list)

        for size in range(1, self.max_size_gram + 1):
            for tokenSeqStr in tokenSeqStrings:
                tokens = tokenSeqStr.split()
                candidate_terms_by_size[size].extend(
                    [" ".join(gram) for gram in ngrams(tokens, size)] * tokenSeqCounter[tokenSeqStr])

            # we need all ngrams (with duplicates) only to extract the frequence
            # for the we will keep a list of unique ngram instances

        self.candidateTerms, self.candidateTermsCounter = self._order_candidate_terms(
            candidate_terms_by_size)

    def _get_substrings(self, term: str) -> List[str]:
        tokens = term.split()
        token_len = len(tokens)

        substrings = set()
        for i in range(1, token_len):
            for gram in ngrams(tokens, i):
                substrings.add(" ".join(gram))

        return list(substrings)

    def _update_stat_triple(self, substring: str, stat_triples: Dict[str, int], parent_term: str, term_frequences: Counter) -> None:

        if substring in stat_triples.keys():

            if parent_term in stat_triples.keys():
                stat_triples[substring][1] = stat_triples[substring][1] + \
                    (term_frequences[parent_term] -
                     stat_triples[parent_term][1])
            else:
                stat_triples[substring][1] = stat_triples[substring][1] + \
                    term_frequences[parent_term]

            stat_triples[substring][2] += 1

        else:
            f_string = 0 if term_frequences.get(
                substring) is None else term_frequences[substring]
            stat_triples[substring] = [
                f_string, term_frequences[parent_term], 1]

    def _process_substrings(self, candidate_term: str, stat_triples: Dict[str, int], term_frequences: Counter) -> None:
        substrings = self._get_substrings(candidate_term)
        for substring in substrings:
            self._update_stat_triple(
                substring, stat_triples, candidate_term, term_frequences)

    def _computes_c_values(self) -> None:

        self._extract_candidate_terms()

        c_values = []
        stat_triples = dict()

        for candidate_term in self.candidateTerms:

            len_candidate_term = len(candidate_term.split())

            if len_candidate_term == self.max_size_gram:
                c_val = math.log2(len_candidate_term) * \
                    self.candidateTermsCounter[candidate_term]
                c_values.append((c_val, candidate_term))

                self._process_substrings(
                    candidate_term, stat_triples, self.candidateTermsCounter)

            else:
                if candidate_term not in stat_triples.keys():
                    c_val = math.log2(len_candidate_term) * \
                        self.candidateTermsCounter[candidate_term]
                    c_values.append((c_val, candidate_term))
                else:
                    c_val = math.log2(
                        len_candidate_term) * (self.candidateTermsCounter[candidate_term] - (stat_triples[candidate_term][1] / stat_triples[candidate_term][2]))
                    c_values.append((c_val, candidate_term))

                self._process_substrings(
                    candidate_term, stat_triples, self.candidateTermsCounter)

        self.c_values = c_values.sort(
            key=lambda term: candidateTermsCounter[term], reverse=True)


if __name__ == "__main__":

    test_texts = [
        'Toothed lock washers - Type V, countersunk',
        'Taper pin - conicity 1/50',
        'T-head bolts with double nib',
        'Handwheels, DIN 950, case-iron, d2 small, without keyway, without handle, form B-F/A',
        'Dog point hexagon socket set screw',
        'Butterfly valve SV04 DIN BF, actuator PAMS93-size 1/2 NC + TOP',
        'Splined Shafts acc. to DIN 5463 / ISO 14',
        'Grooved pins - Half-length reverse-taper grooved',
        'Rod ends DIN ISO 12240-4 (DIN 648) E series stainless version with female thread, maintenance-free',
        'Palm Grips, DIN 6335, light metal, with smooth blind hole, form C, DIN 6335-AL-63-20-C-PL',
        'Hexagon socket set screws with dog point, DIN EN ISO 4028-M5x12 - 45H',
        'Rivet DIN 661  - Type A - 1,6 x 6',
        'Welding neck flange - PN 400 - DIN 2627 - NPS 150',
        'Step Blocks, DIN 6326, adjustable, with spiral gearing, upper part, DIN 6326-K',
        'Loose Slot Tenons, DIN 6323, form C, DIN 6323-20x28-C',
        'Hexagon nut DIN EN 24036 - M3.5 - St'
    ]

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = c_value_tokenizer(nlp)

    for text in test_texts:
        doc = nlp(text)
        print(doc.text)
        print([t.text for t in doc])
        print([t.shape_ for t in doc])
        print(extract_text_sequences_from_corpus([doc]))
        print()
