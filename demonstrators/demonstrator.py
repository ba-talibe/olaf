import os
from dotenv import load_dotenv

import openai
import spacy

from olaf import Pipeline
from olaf.commons.errors import MissingEnvironmentVariable
from olaf.commons.llm_tools import LLMGenerator, DeepSeekGenerator
from olaf.commons.logging_config import logger
from olaf.commons.prompts import (
    llm_prompt_concept_term_extraction,
    llm_prompt_concept_extraction,
    llm_prompt_relation_extraction,
    llm_prompt_relation_term_extraction,
    llm_prompt_term_enrichment,
)
from olaf.pipeline.pipeline_component.concept_relation_extraction import(
    LLMBasedConceptExtraction,
    LLMBasedRelationExtraction
)
from olaf.pipeline.pipeline_component.candidate_term_enrichment import LLMBasedTermEnrichment



from olaf.data_container import CandidateTerm, Concept, Relation
from olaf.pipeline.pipeline_component.term_extraction import LLMTermExtraction
from olaf.repository.corpus_loader import TextCorpusLoader
from olaf.repository.serialiser import KRJSONSerialiser


# based on 



def create_pipeline() -> Pipeline:
    """Initialise a pipeline.

    Returns
    -------
    Pipeline
        The new pipeline created.
    """
    spacy_model = spacy.load("en_core_web_lg")
    corpus_loader = TextCorpusLoader(
        corpus_path=os.path.join(os.getenv('DATA_PATH'), "demo.txt")
    )
    pipeline = Pipeline(
        spacy_model=spacy_model,
        corpus_loader=corpus_loader
    )
    return pipeline


def add_pipeline_components(pipeline: Pipeline) -> Pipeline:
    """Create pipeline with LLM components.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline into which the components are to be added. 

    Returns
    -------
    Pipeline
        The pipeline updated with new components.
    """
    # deepseek_generator = CustomLLMGenerator()
    deepseek_generator = DeepSeekGenerator()
    llm_cterm_extraction = LLMTermExtraction(
        prompt_template=llm_prompt_concept_term_extraction,
        llm_generator=deepseek_generator
    )
    pipeline.add_pipeline_component(llm_cterm_extraction)

    llm_cterm_enrichment = LLMBasedTermEnrichment(
        prompt_template=llm_prompt_term_enrichment,
        llm_generator=deepseek_generator
    )
    pipeline.add_pipeline_component(llm_cterm_enrichment)

  
    llm_concept_extraction = LLMBasedConceptExtraction(
        prompt_template=llm_prompt_concept_extraction, 
        llm_generator=deepseek_generator)
    pipeline.add_pipeline_component(llm_concept_extraction)

    llm_term_extraction = LLMTermExtraction(
        prompt_template=llm_prompt_relation_term_extraction,
        llm_generator=deepseek_generator
    )
    pipeline.add_pipeline_component(llm_term_extraction)

  

    llm_relation_extraction = LLMBasedRelationExtraction(
        prompt_template=llm_prompt_relation_extraction, 
        llm_generator=deepseek_generator)
    pipeline.add_pipeline_component(llm_relation_extraction)


    return pipeline


def main() -> None:
    """LLM pipeline execution."""
    import time
    pipeline = create_pipeline()
    pipeline = add_pipeline_components(pipeline)
    tic = time.time()
    pipeline.run()
    toc = time.time()
    print(f"Execution time: {toc - tic} seconds")
    kr_serialiser = KRJSONSerialiser()
    kr_serialisation_path = os.path.join(
        os.getenv("DATA_PATH"), "llm_pipeline_kr.json")
    kr_serialiser.serialise(kr=pipeline.kr, file_path=kr_serialisation_path)

    kr_rdf_graph_path = os.path.join(
        os.getenv("DATA_PATH"),  "llm_pipeline_kr_rdf_graph.ttl")
    pipeline.kr.rdf_graph.serialize(kr_rdf_graph_path, format="ttl")

    print(f"Nb concepts: {len(pipeline.kr.concepts)}")
    print(f"Nb relations: {len(pipeline.kr.relations)}")
    print(f"Nb metarelations: {len(pipeline.kr.metarelations)}")
    print(f"The KR object has been JSON serialised in : {kr_serialisation_path}")
    print(f"The KR RDF graph has been serialised in : {kr_rdf_graph_path}")

def my_main():
    pipeline = Pipeline(
        spacy_model=spacy.load("en_core_web_lg"),
        corpus_loader=TextCorpusLoader(
            corpus_path=os.path.join(os.getenv('DATA_PATH'), "demo.txt")
        ),
        pipeline_components=[
            LLMTermExtraction(
                prompt_template=llm_prompt_concept_term_extraction,
                llm_generator=DeepSeekGenerator()
            ),
            LLMBasedConceptExtraction(
                llm_prompt_concept_extraction, DeepSeekGenerator()
            )
        ]
    )

    pipeline.run()
    print(f"Nb concepts: {len(pipeline.kr.concepts)}")
    print(f"Nb relations: {len(pipeline.kr.relations)}")

def test_generator():
    corpus = TextCorpusLoader(
            corpus_path=os.path.join(os.getenv('DATA_PATH'), "demo.txt")
        )
    ct = ["Pizza", "dish", "Italian origin", "round", "flattened base", "leavened wheat-based dough", "tomatoes", "cheese", "ingredients", "wood-fired oven"]
    pipeline = Pipeline(
        spacy_model=spacy.load("en_core_web_lg"),
        corpus_loader=corpus,
        pipeline_components=[
            LLMTermExtraction
            (
                llm_prompt_relation_term_extraction, DeepSeekGenerator()
            ),
            LLMBasedRelationExtraction(
                llm_prompt_relation_extraction, DeepSeekGenerator()
            )
        ]
    )
    pipeline.kr.concepts = set([Concept(label=c) for c in ct])

    pipeline.run()
    print(pipeline.kr.relations)

    
    

if __name__ == "__main__":
    load_dotenv()
    main()
