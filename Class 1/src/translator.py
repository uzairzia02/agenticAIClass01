
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
import os
from dotenv import load_dotenv

load_dotenv()

geminiAPIKey = os.getenv("GEMINI_API_KEY")
if not geminiAPIKey:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=geminiAPIKey,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

translatorAgent = Agent(
    name="Translator",
    instructions="You are a translator agent. Translate text from one language to another.",
)

response = Runner.run_sync(
    translatorAgent,
    input="اس بات کو یقینی بنائیں کہ ماڈیول یا پیکج اس ماحول میں انسٹال ہے جس میں آپ کام کر رہے ہیں۔ اگر ان میں سے کوئی بھی مرحلہ مسئلہ حل نہیں کرتا ہے، تو مجھے اپنے سیٹ اپ کے بارے میں مزید تفصیلات بتائیں (جیسے آپ کی ڈائرکٹری کا ڈھانچہ، ماحولیات وغیرہ)، اور میں آپ کو مزید مسائل کے حل میں مدد کروں گا!' translate to english",
    run_config=config
)

print(response.final_output)

