import os
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

folder_to_llm_queries = "folder_to_llm_queries"
rephrasing_queries = json.load(open(os.path.join(folder_to_llm_queries,"queries.json")))

seed = 109
gpt_model = "gpt-4-turbo"

def rephrase_remove_brands():
  client = openai.OpenAI()
  queries = rephrasing_queries
  
  response_json = []
  for ind, q in enumerate(queries):
    msgs = [
      {"role": "system", "content": "You will be provided with statements, and your task is to convert them to English as spoken by native speakers. Drop brand names and all capital lettered words from the text."},
      {"role": "user", "content": "Find the DIONISIO Armchair club chair."},
      {"role": "assistant", "content": "Find the club chair."},
      {"role": "user", "content": "Find the Collin Small ottoman."},
      {"role": "assistant", "content": "Find the small ottoman."},
      {"role": "user", "content": "Find the Live Edge L Shape 64 Desk."},
      {"role": "assistant", "content": "Find the L shape desk."},
      {"role": "user", "content": "Find the Generalmusic introduces GEM GRP800 digital grand piano."},
      {"role": "assistant", "content": "Find the digital grand piano."},
      ]
    
    msgs.append(q)
    response = client.chat.completions.create(
      model=gpt_model,
      messages=msgs,
      temperature=0,
      max_tokens=256,
      seed=seed
    )
    
    response_json.append({
      "query_sentence": json.dumps(msgs),
      "rephrased_sentence": response.choices[0].message
    })
    
    print(f"{ind+1}::{response.choices[0].message}")
    
  file_name = os.path.join(folder_to_llm_queries, 'responses', "rephrase_remove_brands.json")
  with open(file_name, 'w') as f:
    json.dump(response_json, f)

if __name__ == "__main__":
  
  rephrase_remove_brands()
  