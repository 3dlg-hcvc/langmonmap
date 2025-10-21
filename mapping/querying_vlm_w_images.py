import os
import base64
from openai import OpenAI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import json

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

SYSTEM_INSTRUCTIONS = (
   "You are a vision-language assistant. "
    "You are given a query and a list of groups of images. "
    "Each group contains several images. "
    "For every image in each group, return whether the object in the query is clearly visible. "
    "Respond strictly in JSON with this structure:\n\n"
    "{\n"
    '  "response": {\n'
    '    "group_0": [{0: true_or_false}, {1: true_or_false}, ...],\n'
    '    "group_1": [{0: true_or_false}, {1: true_or_false}, ...],\n'
    "    ...\n"
    "  },\n"
    '  "verdict": [list of group numbers where at least one image is true]\n'
    "}\n\n"
    "- Group numbers are integers (0, 1, 2, …).\n"
    "- Use lowercase booleans (`true` / `false`).\n"
    "- Do not include explanations, only the JSON."
)

class Relation(BaseModel):
    relation: str
    from_agent_view: bool
    rationale: str

def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert a NumPy image array to base64-encoded PNG string."""
    img = Image.fromarray(img_array.astype(np.uint8))  # ensure 0-255 uint8
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def query_vlm_with_images(image_groups: list, query_text: str, model="gpt-5"):
  if len(image_groups) == 0:
      return ""
  
  # clean the query text
  query_text = query_text.lower().replace(".","").replace("go to ","").replace("find ","")
  user_prompt = [{"type": "text", "text": f'Query: "{query_text}"\nEvaluate each group separately.'}]
  for g, group in enumerate(image_groups):
    user_prompt.append({"type": "text", "text": f"Group {g}:"})
    for img_array in group:
        user_prompt.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{numpy_to_base64(img_array)}"}, 
            # "detail": "low",
        })

  completion = client.chat.completions.create(
    model=model,   # vision-capable GPT-5 is better
    messages=[
      {"role": "system", "content": SYSTEM_INSTRUCTIONS},
      {"role": "user", "content": user_prompt},
    ],
    response_format={ "type": "json_object"}
  )
  message = completion.choices[0].message.content
  try:
     message = json.loads(message)
  except:
     pass
  
  return message
