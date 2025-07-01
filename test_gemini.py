import google.generativeai as genai
import os

# Get API key from environment variable or hardcoded for testing
API_KEY = "AIzaSyA6Ag6lYW3XyIrcNLajbymj-KbJqnPu-As"

try:
    print("Testing Gemini API connection...")
    
    # Configure the API
    genai.configure(api_key=API_KEY)
    
    # Initialize the model with the latest version
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    print("Using model: gemini-1.5-pro-latest")
    
    # Test with a simple prompt
    response = model.generate_content("Say 'Hello, World!' in a fun way")
    
    print("\n✅ Success! Response from Gemini:")
    print(response.text)
    
except Exception as e:
    print("\n❌ Error:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    
    # Print more detailed error information if available
    if hasattr(e, 'response'):
        print("\nResponse details:")
        print(e.response)
