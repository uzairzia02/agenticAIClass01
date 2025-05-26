from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
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

#Writing an agent that uses the Gemini model
writer = Agent(
    name="Writer",
    instructions = "You are a writer agent. Generate poem, stories, essay, email etc."
)

response = Runner.run_sync(
    writer,
    input="Write a 2 paragraph essay on Pakistan ",
    run_config=config
)

print(response.final_output)

