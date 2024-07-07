'''
Author    -Aditya Bhatt 8:13 PM 07-07-2024

Objective -
1.Is it possible to create a dashboard when you are given any cleaned dataset.
'''
#Import Library
import os
import mlflow
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv()

#Setup Env Vaiables
os.environ["OPENAI_API_KEY"]=os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"]=os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"]=os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["GOOGLE_API_KEY"]=os.getenv("GEMINI_API_KEY")

data = {
    "country": ["Country A", "Country B", "Country C", "Country D"],
    "gdp": [1000, 2000, 3000, 4000],
    "year": [2020, 2021, 2022, 2023],
    "population": [1000000, 2000000, 3000000, 4000000]
}

# Create a DataFrame
df = pd.DataFrame(data)

llm_openai = AzureChatOpenAI(
    deployment_name="test_4o",
    model_name="gpt-4o"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class python coding Expert."),
    ("user", "{input}")
])

llm_google = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
chain_google = prompt | llm_google
chain_azure=prompt | llm_openai
prefix= '''I have a pandas DataFrame 'df' with columns country, gdp, year, and population'''


suffix='''Return only the Python code snippet in Markdown format. The output should be directly executable in a Python interpreter.
'''

query="Plot a chart to show how population has changed with time."

input_query=prefix+"Return code for the following question: "+query+suffix
res_google=chain_google.invoke({"input": input_query})
res_azureopenai=chain_azure.invoke({"input": input_query})

tool = PythonAstREPLTool(locals={"df": df})
print(res_azureopenai.content)
print(res_google.content)

print("Results from Azure OPENAI...")
print("\n")
tool.invoke(res_azureopenai.content)

print("\n")
print("Results from Google...")
tool.invoke(res_google.content)
