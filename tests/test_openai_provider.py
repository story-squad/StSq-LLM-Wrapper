import pytest
import os


def test_create_wrapper():
    from src import LLMWrapper
    API_KEY = os.getenv("OPENAI_API_KEY")
    my_llm = LLMWrapper("openai", API_KEY)
    assert my_llm.API_KEY == API_KEY
    assert my_llm.authenticated
    assert type(my_llm.models) is list
    my_llm = LLMWrapper("openai")
    assert my_llm.API_KEY == API_KEY
    assert my_llm.authenticated
    assert type(my_llm.models) is list
