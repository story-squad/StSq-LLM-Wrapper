import dataclasses


# this class organizes the data received from the llm api in a manner that the consumer of the wrapper can rely on.
@dataclasses.dataclass
class LLMResponse:
    """

    """
    raw_response: 'typing.Any' = object()
    completion: str = None


@dataclasses.dataclass
class LLMDefaults():
    default_completion_model_name: str = "NONE"
    default_search_model_name: str = "NONE"


class OpenaiLLMDefaults():
    default_completion_model_name: str = "text-ada-001"
    default_search_model_name: str = "search-ada-001"


@dataclasses.dataclass
class LLMRequest:
    API_NAME: str = "NONE"
    temperature: float = .5
    max_tokens: int = 40
    top_p: float = .7
    best_of: int = 1
    frequency_penalty: float = .2
    presence_penalty: float = 0
    stop: str = "."
    prompt: str = None
    context: str = None


@dataclasses.dataclass
class OpenaiKWArgs(LLMRequest):
    API_NAME: str = "openai"
    temperature: float = .5
    max_tokens: int = 40
    top_p: float = .7
    best_of: int = 1
    frequency_penalty: float = .2
    presence_penalty: float = 0
    stop: str = "."
    prompt: str = None


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

        if API_NAME.lower() == "openai":
            self.is_openai_api = True
            # set default values for openai api
            self.set_defaults()
            # get the api key from the environment variable if it is not provided
            if not API_KEY:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                openai.api_key = API_KEY
            # get the list of models
            self.models = openai.Model.list()["data"]
            self.API_KEY = openai.api_key
            self.authenticated = True

            if completion_model_name: self.completion_model_name = completion_model_name
            if search_model_name: self.search_model_name = search_model_name
        else:
            raise Exception("Invalid API name")

    def set_defaults(self):
        # set the default values for the openai api
        # TODO: sure there is a programmatic way to do this
        if self.is_openai_api:
            if not self.completion_model_name:
                self.completion_model_name = OpenaiLLMDefaults.default_completion_model_name
            if not self.search_model_name:
                self.search_model_name = OpenaiLLMDefaults.default_search_model_name

    def handle_kwargs(self, request: LLMRequest)-> dict:
        """
        returns kwargs modified to be compatible with the current api
        :rtype: dict
        """
        incoming_class = request.__class__
        if not incoming_class == LLMRequest:
            raise Exception("incoming class is not LLMRequest")

        if self.is_openai_api:
            oai_kwargs = {}
            oai_kwargs["API_NAME"] = "openai"

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
                oai_kwargs["prompt"] = + request.prompt+request.context
            else:
                oai_kwargs["prompt"] = request.prompt

            return oai_kwargs


    def completion(self, prompt=None, kwargs: LLMRequest = None) -> LLMResponse:
        """
        returns the completion response from the llm api
        :param prompt:
        :param kwargs:
        :return:
        """
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
            kwargs_dict = {}

            kwargs = self.handle_kwargs(kwargs)

            # check for compatible kwargs
            if issubclass(kwargs.__class__, LLMRequest):
                if kwargs.API_NAME.lower() != "openai":
                    raise Exception("keyword args class not for use with openai api")

            # if kwargs is not compatible with the current api, return a compatible kwargs

            kwargs.pop("API_NAME")

            model_to_use = self.completion_model_name

            result = openai.Completion.create(model=model_to_use,
                                              **kwargs_dict)
            out_result = LLMResponse(raw_response=result,
                                     completion=result["choices"][0]["text"])

            return out_result

        elif self.is_other:
            raise Exception("not implemented")
