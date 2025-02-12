# Install all the necessary libraries and tools and save the file as stock_crypto_agent.py

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.tavily import TavilyTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from tavily import TavilyClient

# Step 1. Instantiating your TavilyClient
tavily_client = TavilyClient(api_key="TAVILY_API_KEY")



from dotenv import load_dotenv
import os

# Loading a virtual environment with all the dependencies
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")




## Search Web Agent
search_web_agent = Agent(
    name= 'Search Web Agent',
    role= 'Search the web for detailed information',
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo(),TavilyTools()],
    instructions=['Always include sources'],
    show_tool_calls=True,
    markdown=True,
)

## Conventional/ Traditional Stocks Agent
stock_agent = Agent(
    name= " Stocks AI Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,company_news=True,historical_prices=True,technical_indicators=True)],
    show_tool_calls=True,
    description="You are an investment stock analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Use tables to display the data"],
    markdown=True,
)
## Cryptocurrency AI Agent
crypto_agent = Agent(
    name= " Cryptocurrency AI Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,company_news=True,historical_prices=True,technical_indicators=True)],
    show_tool_calls=True,
    description="You are an investment cryptocurrency analyst that researches cryptocurrency prices, analyst recommendations,historical_prices and cryptocurrency fundamentals.",
    instructions=["Use tables to display the data","Use reliable cryptocurrency website to get the latest data"],
    markdown=True,
)
## Multi AI Agent that coordinates and integrates the functionalities of the three agents (web search agent, stock and cryptocurrency agent.
multi_ai_agent = Agent(
    team=[search_web_agent,stock_agent,crypto_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=['Always include sources','Use table to display data'],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize 7 year percentage returns for Binance coin BNB-USD and NVIDIA",stream=True)