from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo 
from dotenv import load_dotenv
import os 
import phi 

from phi.playground import Playground, serve_playground_app
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.1-8b-instant"),  # Changed model
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,  # Fixed typo: was show_tools_calls
    markdown=True
)

# Financial agent
financial_agent = Agent(
    name="Financial Agent",
    role="Provide financial information and analysis",
    model=Groq(id="llama-3.1-8b-instant"),  # Changed model
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

app = Playground(agents = [financial_agent, web_search_agent]).get_app()
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)


