from langchain_core.messages import HumanMessage
import requests
from langchain_core.tools import tool
import os
import sys
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pinecone import Pinecone


load_dotenv()


@tool
def get_courses_schedule_from_api():
    """Get vipassana courses schedule. Если спрашивают, какое расписание на курсе, то этот инструмент не должен вызываться. This tool schould be called when courses schedule is requested. This tool should not be called if a user requests course (singular) schedule."""
    url = "https://seahorse-app-db78s.ondigitalocean.app/api/courses"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None  # Return None in case of an error


@tool
def get_answer_from_document(message: str) -> str:
    """Gets answer from meditation children course document. The tool should receive the whole user message in 'message' parameter. This tool should not be called if a user requests courses (plural) schedule. """
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    # Define the index and namespace
    index_name = "children-courses-org"
    namespace = "children-courses-org"

    # Initialize the embeddings
    model_name = 'multilingual-e5-large'
    embeddings = PineconeEmbeddings(
        model=model_name,
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )

    # Load the existing vector store
    docsearch = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    # Initialize the retriever
    retriever = docsearch.as_retriever()

    # Initialize the language model
    llm = ChatOpenAI(
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        model_name='gpt-4o-mini',
        temperature=0.0
    )

    # Create the retrieval chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    answer_with_knowledge = retrieval_chain.invoke({"input": message})

    return answer_with_knowledge['answer']

# res = get_answer_from_document('скинь ссылку на письмо домой')
# print(res)


tools = [get_answer_from_document, get_courses_schedule_from_api]

tools_mapping = {
    "get_answer_from_document": get_answer_from_document,
    "get_courses_schedule_from_api": get_courses_schedule_from_api
}

llm = ChatOpenAI(model="gpt-4o-mini")


# Usage:
# python chat-playground.py "скинь ссылку на письмо домой"
# python chat-playground.py "какое расписание на детском курсе"
# python chat-playground.py "пришли расписание курсов"
query = sys.argv[1]

print('query', query)
print('\n\n\n')

messages = [HumanMessage(query)]

llm_with_tools = llm.bind_tools(tools, tool_choice='required')

ai_msg = llm_with_tools.invoke(messages)

messages.append(ai_msg)


print('tool calls', ai_msg.tool_calls)
print('\n\n\n')

for tool_call in ai_msg.tool_calls:
    selected_tool = tools_mapping.get(tool_call["name"].lower())
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)


res = llm.invoke(messages)

print('res variable', res)

print('\n\n\n')
print('final answer', res.content)
print('\n\n\n')
