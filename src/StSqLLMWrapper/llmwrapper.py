import dataclasses


@dataclasses.dataclass
class LLMDefaults():
    default_completion_model_name: str = "NONE"


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

    def completion(self, text):
        if self.is_openai_api:
            import openai
            result = openai.Completion.create(model=self.model_defaults.default_completion_model_name, prompt=text)
            return result.choices[0].text
