import os
import time
import google.generativeai as genai

class LegalGenerationEngine:
    def __init__(self):
        # Initialize the stable architecture
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_answer(self, query: str, context_chunks: list) -> str:
        # Assemble your prompt
        context_text = "\n\n".join([chunk.payload.get("document", "") for chunk in context_chunks])
        prompt = f"Using the following legal context, answer the query.\nContext: {context_text}\nQuery: {query}"

        # Stable generation call
        response = self.model.generate_content(prompt)
        
        return response.text
from dotenv import load_dotenv

load_dotenv()

class LegalGenerationEngine:
    def __init__(self):
        # Initialize the Gemini Engine
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_answer(self, query: str, retrieved_chunks: list) -> str:
        # 1. Compile the raw legal text into a readable format for the LLM
        context = ""
        for chunk in retrieved_chunks:
            payload = chunk.payload
            context += f"Act: {payload.get('act')}\nSection: {payload.get('section')}\nText: {payload.get('document')}\n\n"

        # 2. Build the strict prompt guardrails
        prompt = f"""You are NyayaBot, a highly precise Indian Legal AI Assistant.
        Answer the user's question using ONLY the provided legal context below.
        If the exact answer is not contained in the context, explicitly state: "I cannot provide an answer based on the currently verified legal context." Do not hallucinate external knowledge.

        Verified Legal Context:
        {context}

        User Question: {query}
        """

        # 3. Execute the API Call with Smart Retry Logic for Rate Limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's the specific 429 Quota Error
                if "429" in error_msg or "exhausted" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 12 * (attempt + 1) # Wait 12s, then 24s, then 36s
                        print(f"⚠️ Rate limit. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        
                        return "The AI engine is currently experiencing high traffic and rate limits. Please wait one minute and try your question again."
                else:
                    # If it's a completely different error, return it so we can debug
                    return f"An internal error occurred: {str(e)}"
        
        return "Failed to generate an answer after multiple attempts."