
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("API_KEY")

def translate(prompt, language):
  response = openai.Completion.create(engine="text-davinci-003",
    prompt="Translate this into {}\n{}".format(language, prompt),
    temperature=0.3,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0)
   
  if 'choices' in response:
    if len(response["choices"])>0:
      ans = response["choices"][0]["text"]
    else: 
      ans = "Translation Failure"
  else:
    ans = "Translation Failure"

  return ans
