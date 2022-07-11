import pytest
import os


def test_create_wrapper():
    from src import LLMWrapper
    API_KEY = os.getenv("STORYSQUADAI_API_KEY")
    my_llm = LLMWrapper("openai", API_KEY)
    assert my_llm.authenticated
