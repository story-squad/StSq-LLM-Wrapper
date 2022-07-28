import dataclasses



# this class organizes the data received from the llm api in a manner that the consumer of the wrapper can rely on.
@dataclasses.dataclass
class LLMResponse:
    """

    """
    raw_response: 'typing.Any' = object()
    completion: str = None


@dataclasses.dataclass
class LLMDefaults:
    default_completion_model_name: str = None
    default_search_query_model_name: str = None
    default_search_document_model_name: str = None


class OpenaiLLMDefaults():
    default_completion_model_name: str = "text-ada-001"
    default_search_query_model_name: str = "text-search-curie-query-001"
    default_search_document_model_name: str = "curie-search-document"


@dataclasses.dataclass
class LLMRequest:
    API_NAME: str = None
    temperature: float = .5
    max_tokens: int = 40
    top_p: float = .7
    best_of: int = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: str = "."
    prompt: str = None
    context: str = None
    documents: str = None
    query: str = None


@dataclasses.dataclass
class OpenaiKWArgs(LLMRequest):
    API_NAME: str = "openai"
    temperature: float = .5
    max_tokens: int = 40
    top_p: float = .7
    best_of: int = 1
    frequency_penalty: float = .0
    presence_penalty: float = 0
    stop: str = "."
    prompt: str = None


class LLMFilter:
    """Superclass that all LLM filters should inherit from, subclasses should implement the apply() method"""

    def __init__(self, name: str = "unnamed filter"):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __call__(self, request: LLMRequest, response: LLMResponse):
        return self.apply(request, response)

    def apply(self, request: LLMRequest, response: LLMResponse):
        raise NotImplementedError("LLMFilter.apply() not implemented")

    def get_param_value(self, param_name, param_value):
        if param_name in self.params_dict:
            return self.params_dict[param_name].get_value(param_value)
        else:
            raise ValueError("LLMFilter.get_param_value() - Parameter {} not found".format(param_name))

class LLMWrapper:
    def __init__(self, api_name, api_key=None, completion_model_name=None, search_query_model_name=None,
                 search_document_model_name=None):
        """
        :param api_name: openai, or another provider name (only openai in this version)
        :param api_key: provide or leave blank for env variable
        """

        import os
        import openai
        self.completion_model_name = ""
        self.search_model_name = ""
        self.is_openai_api = False

        if api_name.lower() == "openai":
            self.is_openai_api = True
            # set default values for openai api
            self.set_defaults()
            # get the api key from the environment variable if it is not provided
            if not api_key:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                openai.api_key = api_key
            # get the list of models
            self.models = openai.Model.list()["data"]
            self.API_KEY = openai.api_key
            self.authenticated = True

            if completion_model_name: self.completion_model_name = completion_model_name
            if search_query_model_name: self.search_query_model_name = search_query_model_name
            if search_document_model_name: self.search_document_model_name = search_document_model_name

        else:
            raise Exception("Invalid API name")

    def set_defaults(self):
        # set the default values for the openai api
        # TODO: sure there is a programmatic way to do this
        if self.is_openai_api:
            if not self.completion_model_name:
                self.completion_model_name = OpenaiLLMDefaults.default_completion_model_name
            if not self.search_model_name:
                self.search_query_model_name = OpenaiLLMDefaults.default_search_query_model_name
                self.search_document_model_name = OpenaiLLMDefaults.default_search_document_model_name

    def handle_kwargs(self, request: LLMRequest) -> dict:
        """
        returns kwargs modified to be compatible with the current api
        :rtype: dict
        """
        incoming_class = request.__class__
        if not incoming_class == LLMRequest:
            raise Exception("incoming class is not LLMRequest")

        if self.is_openai_api:
            oai_kwargs = {"api_name": "openai"}

            if request.top_p is not None:
                oai_kwargs["top_p"] = request.top_p
            else:
                oai_kwargs["temperature"] = request.temperature

            oai_kwargs["max_tokens"] = request.max_tokens
            oai_kwargs["best_of"] = request.best_of
            oai_kwargs["frequency_penalty"] = request.frequency_penalty
            oai_kwargs["presence_penalty"] = request.presence_penalty
            oai_kwargs["stop"] = request.stop

            if request.context:
                oai_kwargs["prompt"] = request.context + request.prompt
            else:
                oai_kwargs["prompt"] = request.prompt

            return oai_kwargs

    def open_ai_search(self, request: LLMRequest) -> LLMResponse:
        if self.is_openai_api:
            import openai
            import numpy as np

            query_embedding = openai.Embedding.create(input=request.query, model=self.search_query_model_name).data[
                0].embedding
            choices_embeddings = openai.Embedding.create(input=request.documents,
                                                         model=self.search_document_model_name).data
            emb_tup = [
                (choice, choice_emb.embedding) for choice, choice_emb in zip(request.documents, choices_embeddings)]

            def cos_sim(a, b):
                a = np.array(a)
                b = np.array(b)
                return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

            res = [(cos_sim(query_embedding, choice_emb), choice) for choice, choice_emb in emb_tup]
            res = sorted(res, key=lambda x: x[0], reverse=True)
            return res

    def search(self, request: LLMRequest) -> LLMResponse:
        """
        returns the completion response from the llm api
        :param request:
        :param prompt:
        :param kwargs:
        :return:
        """

        if self.is_openai_api:
            import openai

            kwargs = self.handle_kwargs(request)

            if not issubclass(request.__class__, LLMRequest):
                raise Exception("Searches only possible with LLMRequest")
            else:
                if not kwargs["api_name"].lower() == "openai":
                    raise Exception("keyword args class not for use with openai api")
                else:
                    result = self.open_ai_search(request)

                    return result

        elif self.is_other:
            raise Exception("not implemented")

    def completion(self, prompt=None, kwargs: LLMRequest = None) -> LLMResponse:

        # ensure that the prompt is included in the kwargs object for consistency and create a new kwargs object if
        # one is not provided

        if not prompt and not kwargs:
            raise Exception("No kwargs provided")

        if kwargs:
            if issubclass(kwargs.__class__, LLMRequest) and (prompt is not None):
                if kwargs.prompt is not None:
                    raise Exception("Prompt already provided")
                else:
                    kwargs.prompt = prompt
        else:
            kwargs = LLMRequest(prompt=prompt)
            prompt = None

        if self.is_openai_api:
            import openai

            kwargs = self.handle_kwargs(kwargs)

            # check for compatible kwargs
            if issubclass(kwargs.__class__, LLMRequest):
                if kwargs.API_NAME.lower() != "openai":
                    raise Exception("keyword args class not for use with openai api")

            # if kwargs is not compatible with the current api, return a compatible kwargs

            kwargs.pop("api_name")

            model_to_use = self.completion_model_name

            result = openai.Completion.create(model=model_to_use,
                                              **kwargs)
            out_result = LLMResponse(raw_response=result,
                                     completion=result["choices"][0]["text"])

            return out_result

        elif self.is_other:
            raise Exception("not implemented")
