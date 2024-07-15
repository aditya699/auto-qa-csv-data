'''
Author    - Aditya Bhatt 8:13 PM 07-07-2024

Objective -
1. Create a tool that will automate dashboard generation using LLM's.
'''

import os
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv()

# Setup Env Variables
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

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
chain_azure = prompt | llm_openai

prefix = '''I have a pandas DataFrame 'df' with columns country, gdp, year, and population. '''
suffix = '''Return only the Python code snippet in Markdown format (use pandas and matplotlib). The output should be directly executable in a Python interpreter and rendered as a Streamlit output like:
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,8))
# ...
st.pyplot(fig) # instead of plt.show()
'''

def main():
    st.title("CSV Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about the data"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        input_query = prefix + "Return code for the following question: " + prompt + suffix
        
        # Get responses from Google Gemini and Azure OpenAI
        res_google = chain_google.invoke({"input": input_query})
        res_azureopenai = chain_azure.invoke({"input": input_query})

        # Display assistant responses in chat message containers
        with st.chat_message("assistant"):
            st.markdown("### Google Gemini Response:")
            st.markdown(res_google.content)
            
            st.markdown("### Azure OpenAI Response:")
            st.markdown(res_azureopenai.content)

            # Execute the code and display results
            tool = PythonAstREPLTool(locals={"df": df})
            
            st.markdown("### Execution Results:")
            try:
                exec_code = tool.invoke(res_azureopenai.content)
                st.success("Azure OpenAI code executed successfully")
            except Exception as e:
                st.error(f"Error executing Azure OpenAI code: {e}")
            
            try:
                exec_code = tool.invoke(res_google.content)
                st.success("Google Gemini code executed successfully")
            except Exception as e:
                st.error(f"Error executing Google Gemini code: {e}")

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Google Gemini Response:\n{res_google.content}\n\nAzure OpenAI Response:\n{res_azureopenai.content}"
        })

if __name__ == "__main__":
    main()
