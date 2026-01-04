"""Simple API test script"""
import sys
sys.path.insert(0, '/home/wan/onxruntime_genai_api_server')

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test 1: Health check
print("Test 1: Health check")
response = client.get("/")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

# Test 2: List models
print("Test 2: List models")
response = client.get("/v1/models")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

# Test 3: Chat completion
print("Test 3: Chat completion")
response = client.post(
    "/v1/chat/completions",
    json={
        "model": "phi-3.5-mini",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

# Test 4: Invalid model
print("Test 4: Invalid model (should return 404)")
response = client.post(
    "/v1/chat/completions",
    json={
        "model": "invalid-model",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

print("All tests completed!")
