import spacy
import unittest

from commons.ontology_learning_schema import CandidateTerm
from data_preprocessing.data_preprocessing_methods.token_selectors import select_on_pos,select_on_occurrence_count
from term_extraction.term_extraction_methods.c_value import Cvalue
from term_extraction.term_extraction_service import Term_Extraction

class TestOnPosTermExtraction(unittest.TestCase):
    """Test term extraction based on pos tagging.

    Parameters
    ----------
    unittest : TestCase
        Tool for testing
    """

    @classmethod
    def setUpClass(self) -> None:
        corpus = [
            "La réunion de groupe est terminée.",
            "Je voudrais ajouter des participants au groupe.",
            "Les options sont configurables dans les onglets de la fenêtre principale."
        ]
        spacy_model = spacy.load("fr_core_news_sm")
        self.test_spacy_doc = []
        for spacy_document in spacy_model.pipe(corpus):
            self.test_spacy_doc.append(spacy_document)
        
        self.doc_attribute_name = "selected_tokens_4_test"
        spacy_model.add_pipe("token_selector", last=True, config={
            "token_selector_config": {
                "pipeline_name": "test_pipeline",
                "token_selector_names": ["filter_punct", "filter_num", "filter_url","filter_stopwords"],
                "doc_attribute_name": self.doc_attribute_name,
                'make_spans': False,
            }
        })

        self.test_spacy_custom_doc = []
        for spacy_document in spacy_model.pipe(corpus):
            self.test_spacy_custom_doc.append(spacy_document)
        
        self.pos_selection = "NOUN"

    def test_doc_attribute_selection(self):
        """Test the _get_doc_content_for_term_extraction method to know if the correct attribute is returned depending on the doc_attribute_name definition and value.
        """
        config = {
            "selected_tokens_doc_attribute": None,
            "use_span" : False,
            "on_pos": {
                "pos_selection": ["NOUN"],
                "use_lemma": False
            }
        }
        config_custom = {
            "selected_tokens_doc_attribute": self.doc_attribute_name,
            "use_span" : False,
            "on_pos": {
                "pos_selection": ["NOUN"],
                "use_lemma": True
            }
        }
        term_extraction_instance = Term_Extraction(self.test_spacy_doc, config)
        term_extraction_instance_custom = Term_Extraction(self.test_spacy_custom_doc, config_custom)
        self.assertEqual(term_extraction_instance_custom._get_doc_content_for_term_extraction(self.doc_attribute_name, self.test_spacy_custom_doc[0]), self.test_spacy_custom_doc[0]._.get(self.doc_attribute_name))
        self.assertEqual(term_extraction_instance._get_doc_content_for_term_extraction(None, self.test_spacy_doc[0]), self.test_spacy_doc[0])


    def test_on_pos_results(self):
        """Test the on_pos_term_extraction function to check that correct terms are extracted depending on pos tags.
        """
        corpus_noun = [
            "réunion", "groupe", "participants", "options", "onglets", "fenêtre"
        ]
        corpus_noun_ct = set()
        for noun in corpus_noun : 
            corpus_noun_ct.add(CandidateTerm(noun))

        corpus_noun_lemma = {
            "réunion", "groupe", "participant", "option", "onglet", "fenêtre"
        }
        corpus_noun_lemma_ct = set()
        for noun in corpus_noun_lemma:
            corpus_noun_lemma_ct.add(CandidateTerm(noun))

        config = {
            "selected_tokens_doc_attribute": None,
            "use_span" : False,
            "on_pos": {
                "pos_selection": ["NOUN"],
                "use_lemma": False
            }
        }
        config_lemma = {
            "selected_tokens_doc_attribute": None,
            "use_span" : False,
            "on_pos": {
                "pos_selection": ["NOUN"],
                "use_lemma": True
            }
        }
        term_extraction_instance = Term_Extraction(self.test_spacy_doc, config)
        term_extraction_instance_lemma = Term_Extraction(self.test_spacy_doc, config_lemma)
        self.assertEqual(corpus_noun_ct,set(term_extraction_instance.on_pos_term_extraction()))
        self.assertEqual(corpus_noun_lemma_ct,set(term_extraction_instance_lemma.on_pos_term_extraction()))

        term_extraction_instance.config["use_span"] = True
        self.assertListEqual(term_extraction_instance.on_pos_term_extraction(), [])

class TestOnoccurrenceTermExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        corpus = [
            "La réunion de groupe est terminée, je voudrais ajouter des participants au groupe principal pour les prochaines réunions et les options de groupe sont configurables dans les onglets de la fenêtre principale."
        ]
        spacy_model = spacy.load("fr_core_news_sm")
        self.doc_attribute_name = "selected_tokens_4_test"
        spacy_model.add_pipe("token_selector", last=True, config={
            "token_selector_config": {
                "pipeline_name": "test_pipeline",
                "token_selector_names": ["filter_punct", "filter_num", "filter_url","filter_stopwords"],
                "doc_attribute_name": self.doc_attribute_name,
                'make_spans': False,
            }
        })
        
        self.test_spacy_doc = []
        for spacy_document in spacy_model.pipe(corpus):
            self.test_spacy_doc.append(spacy_document)

    def test_on_occurrence_results(self):
        config = {
            "selected_tokens_doc_attribute": self.doc_attribute_name,
            "on_occurrence": {
                "occurrence_threshold": 1,
                "use_lemma": False
            }
        }
        config_lemma = {
            "selected_tokens_doc_attribute": self.doc_attribute_name,
            "on_occurrence": {
                "occurrence_threshold": 1,
                "use_lemma": True
            }
        }

        term_extraction_instance = Term_Extraction(self.test_spacy_doc, config)
        term_extraction_instance_lemma = Term_Extraction(self.test_spacy_doc, config_lemma)

        words_occurrence_selection = {"groupe"}
        words_occurrence_selection_ct = set()
        for word in words_occurrence_selection:
            words_occurrence_selection_ct.add(CandidateTerm(word))

        lemmas_occurrence_selection = {"groupe", "réunion", "principal"}
        lemmas_occurrence_selection_ct = set()
        for lemma in lemmas_occurrence_selection:
            lemmas_occurrence_selection_ct.add(CandidateTerm(lemma))

        self.assertEqual(set(term_extraction_instance.on_occurrence_term_extraction()), words_occurrence_selection_ct)
        self.assertEqual(set(term_extraction_instance_lemma.on_occurrence_term_extraction()), lemmas_occurrence_selection_ct)
        

class TestCvalue(unittest.TestCase):
    """Test the C-value computation according to the examples in <https://doi.org/10.1007/s007999900023> (section 2.3.1, page 5).
    """

    @classmethod
    def setUpClass(self):
        self.test_terms = [
            "ADENOID CYSTIC BASAL CELL CARCINOMA",
            "CYSTIC BASAL CELL CARCINOMA",
            "ULCERATED BASAL CELL CARCINOMA",
            "RECURRENT BASAL CELL CARCINOMA",
            "CIRCUMSCRIBED BASAL CELL CARCINOMA",
            "BASAL CELL CARCINOMA"
        ]

        test_terms_counter = {
            "ADENOID CYSTIC BASAL CELL CARCINOMA": 5,
            "CYSTIC BASAL CELL CARCINOMA": 11,
            "ULCERATED BASAL CELL CARCINOMA": 7,
            "RECURRENT BASAL CELL CARCINOMA": 5,
            "CIRCUMSCRIBED BASAL CELL CARCINOMA": 3,
            "BASAL CELL CARCINOMA": 984
        }

        vocab_strings = []
        for term in self.test_terms:
            vocab_strings.extend(term.split())

        vocab = spacy.vocab.Vocab(strings=vocab_strings)

        self.doc_attribute_name = "cvalue_token_sequences"

        if not spacy.tokens.doc.Doc.has_extension(self.doc_attribute_name):
            spacy.tokens.doc.Doc.set_extension(
                self.doc_attribute_name, default=[])

        test_terms_spans = []
        corpus = []
        for term in self.test_terms:
            words = term.split()
            spaces = [True] * len(words)
            doc = spacy.tokens.Doc(vocab, words=words, spaces=spaces)
            doc._.set(self.doc_attribute_name, [doc[:]])
            test_terms_spans.append(doc[:])
            corpus.append(doc)

        my_c_val = Cvalue(
            corpus=corpus, tokenSequences_doc_attribute_name=self.doc_attribute_name, max_size_gram=5)

        # we manually set the candidate terms and their occurencies otherwise the process considers all
        # the ngrams extracted from the terms. This is not done like this in the paper.
        my_c_val.candidateTermsSpans = test_terms_spans
        my_c_val.candidateTermsCounter = test_terms_counter

        self.c_values = my_c_val()

    def test_Cvalue_results(self):
        self.assertEqual(len(self.c_values), len(set(self.test_terms)))

        self.assertEqual(round(self.c_values[0].c_value, 2), 1551.36)
        self.assertEqual(
            self.c_values[0].candidate_term, "BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[1].c_value), 14.0)
        self.assertEqual(
            self.c_values[1].candidate_term, "ULCERATED BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[2].c_value), 12.0)
        self.assertEqual(
            self.c_values[2].candidate_term, "CYSTIC BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[3].c_value, 4), 11.6096)
        self.assertEqual(self.c_values[3].candidate_term,
                         "ADENOID CYSTIC BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[4].c_value), 10.0)
        self.assertEqual(
            self.c_values[4].candidate_term, "RECURRENT BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[5].c_value), 6.0)
        self.assertEqual(self.c_values[5].candidate_term,
                         "CIRCUMSCRIBED BASAL CELL CARCINOMA")


if __name__ == '__main__':
    unittest.main()
