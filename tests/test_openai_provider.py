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


def test_completion_with_specific_api_kwargs():
    from src import LLMWrapper
    from src import OpenaiKWArgs
    kws = OpenaiKWArgs(temperature=1,
                       max_tokens=40,
                       top_p=.7,
                       best_of=1,
                       frequency_penalty=.2,
                       presence_penalty=0,
                       stop=".")
    API_KEY = os.getenv("OPENAI_API_KEY")
    my_llm = LLMWrapper("openai", API_KEY)
    result = my_llm.completion(prompt="hello how are you?", kwargsclass=kws)
    assert type(result) is str


def test_completion_with_invalid_api_kwargs():
    from src import LLMWrapper
    from src import KWArgs_class
    kws = KWArgs_class()
    API_KEY = os.getenv("OPENAI_API_KEY")
    my_llm = LLMWrapper("openai", API_KEY)
    with pytest.raises(Exception) as e_info:
        result = my_llm.completion(prompt="hello how are you?", kwargsclass=kws)
    assert "keyword args class not for use with openai api" in str(e_info)
