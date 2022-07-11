import pytest
def test_create_wrapper():
    from src import LLMWrapper
    print(dir(LLMWrapper))