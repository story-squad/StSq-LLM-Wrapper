class LLMWrapper:
    def __init__(self, API_NAME, API_KEY=None):
        import os
        import openai
        if API_NAME.lower() == "openai":
            if not API_KEY:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            self.models = openai.Model.list()
            self.authenticated = True



