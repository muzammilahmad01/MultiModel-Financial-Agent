from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo 
from dotenv import load_dotenv

load_dotenv()

# Set your Groq API key


# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),  # Changed model
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,  # Fixed typo: was show_tools_calls
    markdown=True
)

# Financial agent
financial_agent = Agent(
    name="Financial Agent",
    role="Provide financial information and analysis",
    model=Groq(id="llama-3.3-70b-versatile"),  # Changed model
    tools=[YFinanceTools(
        stock_price=True,
        company_info=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=["Use tables to display data"],
    show_tool_calls=True,  # Fixed typo
    markdown=True,
)

# Multi-agent coordinator
multi_ai_agents = Agent(
    team=[web_search_agent, financial_agent],
    model=Groq(id="llama-3.3-70b-versatile"),  # Add model to coordinator
    instructions=["Collaborate to provide accurate and comprehensive information"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agents.print_response(
    "Supply a detailed analysis of Apple's (AAPL) current stock performance including recent news, stock price trends, and analyst recommendations.", 
    stream=True
)