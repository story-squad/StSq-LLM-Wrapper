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


def test_create_wrapper_with_invalid_api_name():
    from src import LLMWrapper
    with pytest.raises(Exception) as e_info:
        LLMWrapper("invalid_api_name")
    assert "Invalid API name" in str(e_info.value)


def test_create_wrapper_with_invalid_api_key():
    from src import LLMWrapper
    with pytest.raises(Exception) as e_info:
        LLMWrapper("openai", "invalid_api_key")
    assert "Invalid API key" in str(e_info.value) or "Incorrect API key" in str(e_info.value)


def test_default_completion_call():
    from src import LLMWrapper
    API_KEY = os.getenv("OPENAI_API_KEY")
    my_llm = LLMWrapper("openai", API_KEY)
    result = my_llm.completion("hello how are you?")
    assert type(result) is str

