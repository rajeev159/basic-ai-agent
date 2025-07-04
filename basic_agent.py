import os

# from langchain_openai import  ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.tools import Tool

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
                             google_api_key=GOOGLE_API_KEY)

# Option 1: DuckDuckGo Search (community tool, usually no API key needed)
search_tool = DuckDuckGoSearchRun()

tools = [search_tool]

# Add calculator
math_tool = Tool(name="Calculator", 
                 func=LLMMathChain.from_llm(llm).run, 
                 description="Useful for when you need to answer "
                 "questions about math.")

tools.append(math_tool)

# The 'react' prompt is a common and effective prompt for agents.
prompt = hub.pull("hwchase17/react")

# create_react_agent helps us set up the agent logic with the LLM, 
# tools, and prompt.
agent = create_react_agent(llm, tools, prompt)

# The AgentExecutor is responsible for running the agent, handling tool calls,
# and managing the interaction loop.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("--- How can assist you today? ---")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"\nAgent: {response['output']}")
    except Exception as e:
        print(f"An error occurred: {e}")

print("\n--- Agent conversation ended. ---")