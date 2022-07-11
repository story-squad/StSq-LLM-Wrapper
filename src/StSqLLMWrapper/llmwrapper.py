import dataclasses


@dataclasses.dataclass
class LLMDefaults():
    default_completion_model_name: str = "NONE"


@dataclasses.dataclass
class KWArgs_class:
    API_NAME: str = "NONE"


@dataclasses.dataclass
class OpenaiKWArgs(KWArgs_class):
    API_NAME: str = "openai"
    temperature: int = 50,
    max_tokens: int = 40,
    top_p: float = .7,
    best_of: int = 1,
    frequency_penalty: float = .2,
    presence_penalty: float = 0,
    stop: str = "."


class LLMWrapper:
    def __init__(self, API_NAME, API_KEY=None):
        import os
        import openai
        if API_NAME.lower() == "openai":
            self.model_defaults = LLMDefaults(default_completion_model_name="text-ada-001")
            if not API_KEY:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                openai.api_key = API_KEY
            self.models = openai.Model.list()["data"]
            self.authenticated = True
            self.API_KEY = openai.api_key
            self.is_openai_api = True
        else:
            raise Exception("Invalid API name")

    def completion(self, prompt="test", kwargsclass: KWArgs_class = {}):
        if self.is_openai_api:
            import openai
            kwargs_dict= {}
            if issubclass(kwargsclass.__class__,KWArgs_class) :
                if kwargsclass.API_NAME.lower() != "openai":
                    raise Exception("keyword args class not for use with openai api")
                kwargs_dict = kwargsclass.__dict__
                kwargs_dict.pop("API_NAME")
            result = openai.Completion.create(model=self.model_defaults.default_completion_model_name, prompt=prompt,
                                              **kwargs_dict)
            return result.choices[0].text
