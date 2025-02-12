import openai
import phi.api
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.tavily import TavilyTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from tavily import TavilyClient
from dotenv import load_dotenv 

# Set the Phidata Playground for interaction of the agents
import phi
import os
from phi.playground import Playground,serve_playground_app


# Load the environment variables
load_dotenv()

# Playground authentication key to connect to the endpoint in localhost
phi.api = os.getenv("PHI_API_KEY")

## Search Web Agent for searching the internet for information

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

my_app=Playground(agents=[search_web_agent,stock_agent,crypto_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:my_app",reload=True)
   

