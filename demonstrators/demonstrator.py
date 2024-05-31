import re
import spacy

from olaf import Pipeline
from olaf.pipeline.pipeline_component.term_extraction import POSTermExtraction
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    CTsToRelationExtraction,
)
from olaf.pipeline.pipeline_component.axiom_extraction.owl_axiom_extraction import (
    OWLAxiomExtraction,
)
from olaf.data_container.knowledge_representation_schema import KnowledgeRepresentation
from olaf.repository.serialiser import BaseOWLSerialiser
from olaf.repository.corpus_loader.text_corpus_loader import TextCorpusLoader
from olaf.pipeline.pipeline_component.term_extraction.tfidf_term_extraction import (
    TFIDFTermExtraction,
)
from olaf.pipeline.pipeline_component.concept_relation_extraction.agglomerative_clustering_concept_extraction import (
    AgglomerativeClusteringConceptExtraction,
)


def process_apostrophe(text):
    pattern1 = r"(\w+)'(\w+)"
    pattern2 = r"(\w+) '(\w+)"

    replacement = r"\1' \2"

    result = re.sub(pattern1, replacement, text)
    result = re.sub(pattern2, replacement, result)
    return result


if __name__ == "__main__":
    spacy_sm_model = spacy.load("fr_core_news_sm")  # load a small size french model
    spacy_md_model = spacy.load("fr_core_news_md")  # load a meduim size french model

    # corpus_path = '/content/drive/MyDrive/dataset/Documentation_imprimante_page_67.txt'
    corpus_path = "demonstrators/Documentation_imprimante_page_67.txt"
    corpus = TextCorpusLoader(corpus_path)._read_corpus()
    corpus = [doc[:-1] for doc in corpus]
    aggr_corpus = " ".join(corpus)  # corpus totaly joined
    aggr_corpus = aggr_corpus.split(".")
    aggr_corpus = [process_apostrophe(doc) for doc in aggr_corpus]
    aggr_corpus

    # concept extraction component
    concept_term_extraction = TFIDFTermExtraction(
        **{
            "candidate_term_threshold": 0.5,
            "max_term_token_length": 10,
            "tfidf_agg_type": "MAX",
        }
    )

    concept_extraction = AgglomerativeClusteringConceptExtraction(nb_clusters=3)

    # concept extraction component
    relation_term_extraction = TFIDFTermExtraction(
        **{
            "candidate_term_threshold": 0.35,
            "max_term_token_length": 2,
            "tfidf_agg_type": "MAX",
        }
    )

    relation_extraction = AgglomerativeClusteringConceptExtraction(
        **{
            "distance_threshold": 0.2,
        }
    )

    olaf_pipeline = Pipeline(
        spacy_model=spacy_md_model,
        pipeline_components=[
            concept_term_extraction,
            concept_extraction,
            # relation_term_extraction,
            # relation_extraction,
        ],
        corpus=list(spacy_md_model.pipe(aggr_corpus)),
    )

    olaf_pipeline.run()
