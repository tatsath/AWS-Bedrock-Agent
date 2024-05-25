import boto3
import json

prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    "prompt":"[INST]"+ prompt_data +"[/INST]",
    "max_tokens":1000,
    "temperature":0.7,
    "top_k":50,
    "top_p":0.7
}
body=json.dumps(payload)
model_id="mistral.mistral-7b-instruct-v0:2"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)


response_body=json.loads(response.get("body").read())
outputs = response_body.get("outputs")
completions = [output["text"] for output in outputs]
print(completions)


