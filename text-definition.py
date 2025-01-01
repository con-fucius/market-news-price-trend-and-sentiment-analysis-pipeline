import openai
import anthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.llms import Anthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your API keys
OPENAI_API_KEY = 'your_openai_api_key'  # Replace with your OpenAI API key
ANTHROPIC_API_KEY = 'your_anthropic_api_key'  # Replace with your Anthropic API key

# Initialize OpenAI and Anthropic API clients
openai.api_key = OPENAI_API_KEY
anthropic_client = anthropic.Client(ANTHROPIC_API_KEY)

# Step 1: Define Prompts
definition_prompt = PromptTemplate(
    input_variables=["term"],
    template="Please provide the definition of the term: {term}."
)

synonym_prompt = PromptTemplate(
    input_variables=["term"],
    template="Provide some synonyms for the word: {term}."
)

antonym_prompt = PromptTemplate(
    input_variables=["term"],
    template="Provide some antonyms (opposites) for the word: {term}."
)

# Step 2: Initialize LLMs (OpenAI for definition, Anthropic for synonyms, and OpenAI for antonyms)
openai_llm = OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
anthropic_llm = Anthropic(model="claude-1", anthropic_api_key=ANTHROPIC_API_KEY)

# Step 3: Define Chains for each task (definition, synonyms, antonyms)
definition_chain = LLMChain(prompt=definition_prompt, llm=openai_llm)
synonym_chain = LLMChain(prompt=synonym_prompt, llm=anthropic_llm)
antonym_chain = LLMChain(prompt=antonym_prompt, llm=openai_llm)

# Step 4: Define Main Function to Run the Agent
def get_term_info(term):
    print(f"Fetching information for the term: {term}\n")
    
    # 1. Get definition using OpenAI (GPT-4)
    definition = definition_chain.run(term=term)
    print(f"Definition of {term}:\n{definition}\n")
    
    # 2. Get synonyms using Anthropic (Claude)
    synonyms = synonym_chain.run(term=term)
    print(f"Synonyms of {term}:\n{synonyms}\n")
    
    # 3. Get antonyms using OpenAI (GPT-4)
    antonyms = antonym_chain.run(term=term)
    print(f"Antonyms of {term}:\n{antonyms}\n")

# Test the agent with an example term
term = "happy"
get_term_info(term)
