#!/usr/bin/env python3
"""Quick script to list available Gemini models"""
import google.generativeai as genai

with open('hehe.txt', 'r') as f:
    for line in f:
        if 'KEY =' in line:
            api_key = line.split('=')[1].strip().strip('"')
            break

genai.configure(api_key=api_key)

print("Available Gemini models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")
