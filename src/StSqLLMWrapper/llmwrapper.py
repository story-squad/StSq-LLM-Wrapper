class LLMWrapper:
    def __init__(self, API_NAME, API_KEY=None):
        import os
        import openai
        if API_NAME.lower() == "openai":
            if not API_KEY:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                openai.api_key = API_KEY
            self.models = openai.Model.list()["data"]
            self.authenticated = True
            self.API_KEY = openai.api_key
        else:
            raise Exception("Invalid API name")


