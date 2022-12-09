import spacy.tokens
from typing import Any, Dict, List

from commons.ontology_learning_schema import KR
from config.core import config
from config import logging_config
from relation_extraction.relation_extraction_methods.on_occurrence_relation_extraction import OnOccurrenceRelationExtraction

class RelationExtraction():

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], kr: KR, configuration: Dict[str, Any] = None) -> None:
        self.corpus = corpus
        self.kr = kr
        if configuration is None:
            self.config = config['relation_extraction']
        else:
            self.config = configuration

    def on_occurence_relation_extraction(self) -> None:
        try : 
            assert self.config['on_occurrence'].get('use_lemma') is not None
            assert self.config['on_occurrence'].get('threshold') is not None
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for relation extraction based on co-occurrence. Make sure you provided the configuration fields:
                    - relation_extraction.on_occurrence.use_lemma
                    - relation_extraction.on_occurrence.threshold
                    Trace : {e}
                """)
        else : 
            options = self.config['on_occurrence']
        relation_extration = OnOccurrenceRelationExtraction(self.corpus, self.kr, options)
        relation_extration.on_occurrence_relation_extraction()

