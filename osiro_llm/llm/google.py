import os
import google.generativeai as genai

class GoogleLLMWrapper:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash-lite"):
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API Key must be provided or set as GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_content(self, prompt):
        """
        Generates content from the LLM based on a prompt.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"An error occurred with the LLM API: {e}")
            return "" # Return empty string on error
