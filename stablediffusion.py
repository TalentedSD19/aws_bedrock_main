import boto3
import json
import base64
import os

prompt_data = """
provide me an 4k hd image of a mountain, also use a blue sky snowy season and
mountaineers climbing
"""
prompt_template=[{"text":prompt_data}]
bedrock = boto3.client(service_name="bedrock-runtime")
# payload = {
#     "textToImageParams":prompt_template,
#     "taskType":"TEXT_IMAGE",
#     "imageGenerationConfig":{"cfg_scale": 8,
#     "seed": 42,
#     "quality":"standard",
#     "width":512,
#     "height":512,
#     "numberOfImages":1},
# }
payload2 = {
    "textToImageParams": {
        "text": prompt_data
    },
    "taskType": "TEXT_IMAGE",
    "imageGenerationConfig": {
        "cfgScale": 8,
        "seed": 42,
        "quality": "standard",
        "width": 1024,
        "height": 1024,
        "numberOfImages": 1
    }
}

body = json.dumps(payload2)
model_id = "amazon.titan-image-generator-v1"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
with open("a.txt","w") as a:
    a.write(str(response_body))
# print(response_body)
artifact = response_body["images"][0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save image to a file in the output directory.
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)
