import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Print the API key to verify it's loaded correctly (optional)
print(openai_api_key)

# Now you can use the openai_api_key with OpenAI's API
# For example, if you're using the openai library:
# import openai
# openai.api_key = openai_api_key
