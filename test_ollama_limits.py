#!/usr/bin/env python3
# test_ollama_limits.py
import httpx
import json
import time

def test_ollama_with_size(num_messages=1, message_length=100, system_prompt_length=100):
    """Test Ollama with different message sizes"""
    url = "http://localhost:11434/api/chat"
    
    # Build messages
    messages = []
    
    # System prompt
    if system_prompt_length > 0:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. " + "x" * system_prompt_length
        })
    
    # Add conversation history
    for i in range(num_messages):
        messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: " + "test " * (message_length // 5)
        })
    
    # Add final user message
    messages.append({"role": "user", "content": "Please respond with 'OK'"})
    
    # Calculate approximate size
    total_chars = sum(len(msg['content']) for msg in messages)
    
    payload = {
        "model": "mistral:latest",  # or whatever model you're using
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 50,
            "temperature": 0.1,
            "num_ctx": 2048,
        }
    }
    
    print(f"\n--- Test: {num_messages} messages, ~{total_chars} chars total ---")
    
    try:
        start_time = time.time()
        response = httpx.post(url, json=payload, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('message', {}).get('content', '')
            print(f"✓ Success in {elapsed:.1f}s: {response_text[:50]}...")
            return True
        else:
            print(f"✗ Error {response.status_code}: {response.text[:100]}...")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

# Run tests with increasing complexity
print("Testing Ollama limits...")

# Test 1: Minimal
if test_ollama_with_size(num_messages=1, message_length=50, system_prompt_length=50):
    print("Basic test passed")
else:
    print("Even basic test failed!")
    exit(1)

# Test 2: Small conversation
test_ollama_with_size(num_messages=4, message_length=100, system_prompt_length=100)

# Test 3: Medium conversation
test_ollama_with_size(num_messages=10, message_length=200, system_prompt_length=200)

# Test 4: Large system prompt (like with tools)
test_ollama_with_size(num_messages=2, message_length=100, system_prompt_length=2000)

# Test 5: Many messages
test_ollama_with_size(num_messages=20, message_length=100, system_prompt_length=100)

# Test 6: Long messages
test_ollama_with_size(num_messages=4, message_length=1000, system_prompt_length=100)

print("\nTests complete. Use the last successful configuration as your limit.")