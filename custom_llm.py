import os
from langchain.prompts import PromptTemplate
import requests
from langchain.llms.base import LLM
from typing import Optional, List
# Use the custom model with Langchain's chain system
from langchain.chains import LLMChain


class CustomLLM(LLM):
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        super().__init__()
        # Initialize the endpoint URL and API key
        self._endpoint_url = endpoint_url  # Set the custom API endpoint URL
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")  # Fetch API key from environment variable if not provided

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Override the _call method to send requests to the OpenAI API"""
        # Headers for OpenAI API requests
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

        # Data payload (customize based on your needs)
        data = {
            "model": "gpt-3.5-turbo", 
            "messages": [{"role": "user", "content": prompt}],  # The prompt goes here
            "max_tokens": 500,  # Adjust token limit
            "temperature": 0.5,  # Adjust temperature for creativity
        }

        # Send POST request to the OpenAI API
        response = requests.post(self._endpoint_url, headers=headers, json=data)

        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"Error calling OpenAI API: {response.status_code}, {response.text}")

        # Parse the response JSON and extract the message content
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        """Required property to specify LLM type in Langchain"""
        return "custom_openai_chat"

    @property
    def _identifying_params(self) -> dict:
        """Return the identifying parameters (used by Langchain)"""
        return {"endpoint_url": self._endpoint_url}
    
if __name__=='__main__':
    
    # Initialize the custom LLM with your endpoint URL
    custom_llm = CustomLLM(endpoint_url="https://api.openai.com/v1/chat/completions")

    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="Generate a response for: {input}"
    )

    chain = prompt_template | custom_llm 

    # Run the chain with some input
    result = chain.invoke("What is the capital of France?")
    print(result)