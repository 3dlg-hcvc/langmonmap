import os
import base64, mimetypes, json, sys
from openai import OpenAI
from pydantic import BaseModel

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

SYSTEM_INSTRUCTIONS = (
    " You will infer the most likely relation between object A (colored in red) and object B (colored in blue) from the image. "
    " Note that this is a top-down projection of the image on a map."
    " In some images, you will find a green box, denoting the agent location. "
    " In that case infer the relation of A left or right of B when viewed from the agent location. "
    " Consider the nearest blobs for A and B to the agent."
    "Return ONLY JSON:\n"
    '{\n'
    '  "relation": one of ["left of","right of","in front of","behind","next to","near","on","above","below","overlapping","undetermined"],\n'
    '  "from_agent_view": true|false,  // true iff a green agent box is visible and used\n'
    '  "rationale": "1-2 short phrases"\n'
    '}\n'
    "Rules:\n"
    "- Use Euclidean proximity for near/next to (next to = very close, non-overlapping; near = close but not touching).\n"
    "- 'On' or 'overlapping' if A's red footprint lies mostly within or overlapping B's blue footprint.\n"
    "- If agent is present: decide LEFT/RIGHT relative to the ray from agent toward the midpoint between A and B "
    "(the object appearing on the left side of that ray is 'left of' the other). "
    "- Consider the nearest blobs for A and B to the agent. "
    "Prefer left/right if clearly visible; else fall back to other relations.\n"
    "- Be concise and avoid extra prose."
)

class Relation(BaseModel):
    relation: str
    from_agent_view: bool
    rationale: str

def to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    
completion = client.chat.completions.create(
  model="gpt-5",
  messages=[
    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
    {"role": "user", "content": [{"type": "text", "text": "Infer the spatial relation between red A and blue B. If a green agent box is present, decide LEFT/RIGHT from the agent's viewpoint. Be concise."},
     { "type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{to_data_url('/localhome/sraychau/Projects/Research/LangMultion/langmonmap/center table_near_chair.jpg')}"}}]}
  ],
  response_format={ "type": "json_object"}
)

message = completion.choices[0].message.content
print(f"{message}")
