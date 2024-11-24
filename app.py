from flask import Flask, jsonify, request
from dotenv import load_dotenv
import requests
import os
from bs4 import BeautifulSoup

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pinecone import Pinecone

app = Flask(__name__)

if os.getenv('FLASK_ENV') == 'development':
    load_dotenv()

@app.route('/')
def hello_fly():
    return 'hello world'

@app.route('/api/courses', methods=['GET'])
def get_courses():
    url = "https://www.dhamma.org/ru/schedules/schdullabha"
    response = requests.get(url)
    
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch the page"}), 500
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Use the query selector to find the table body
    table_body = soup.select_one("body > div > div > div:nth-child(8) > div:nth-child(6) > table:nth-child(4) > tbody")
    
    if not table_body:
        return jsonify({"error": "Failed to find the courses table"}), 500
    
    courses = []
    
    # Iterate over all tr elements except the first one (header)
    for tr in table_body.find_all('tr')[1:]:
        tds = tr.find_all('td')

        link = tds[0].find('a', text='Анкета*')
        if link:
            url = link.get('href')
        else:
          url = None
        
        if len(tds) < 6:
            continue

        print(tds[5])
        
        course = {
            "application_url": url,
            "date": tds[1].get_text(strip=True),
            "type": tds[2].get_text(strip=True),
            "status": tds[3].get_text(strip=True),
            "location": tds[4].get_text(strip=True),
            "description": tds[5].get_text(strip=True),
        }
        
        courses.append(course)
    
    return jsonify(courses)

# Replace with your actual tokens

# Is My Car Okay Bot
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_BOT_TOKEN_CHILDREN_COURSES_ORG = os.environ.get('TELEGRAM_BOT_TOKEN_CHILDREN_COURSES_ORG')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL')

@app.route('/webhook', methods=['POST'])
def webhook():
    update = request.get_json()
    if 'message' in update:
        chat_id = update['message']['chat']['id']
        user_message = update['message']['text']
        
        # Define the index and namespace for Pincone vector database
        index_name = "minsk-knowledge"
        namespace = "minsk"
        response = get_answer_from_document(user_message, index_name, namespace)
        
        bot_token = TELEGRAM_BOT_TOKEN 
        send_message(chat_id, response, bot_token)
    
    return '', 200

@app.route('/webhook-children-courses-org', methods=['POST'])
def webhook_children_courses_org():
    update = request.get_json()
    if 'message' in update:
        chat_id = update['message']['chat']['id']
        user_message = update['message']['text']
        
        # Define the index and namespace for Pincone vector database
        index_name = "children-courses-org"
        namespace = "children-courses-org"

        response = get_answer_from_document(user_message,index_name, namespace)

        bot_token = TELEGRAM_BOT_TOKEN_CHILDREN_COURSES_ORG 
        send_message(chat_id, response, bot_token)

        # Send analitics to Slack
        data = {'text': user_message}
        requests.post(SLACK_WEBHOOK_URL, json=data)

    return '', 200

def get_gpt_response(user_message):
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'gpt-3.5-turbo',  # or any other model you want to use
        'messages': [{'role': 'user', 'content': user_message}],
    }
    
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    response_json = response.json()
    
    # Extract the assistant's reply
    if 'choices' in response_json and len(response_json['choices']) > 0:
        return response_json['choices'][0]['message']['content']
    else:
        return "Sorry, I couldn't get a response from the AI."

def send_message(chat_id, text, bot_token):
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown'  # Optional: use Markdown formatting
    }
    requests.post(url, json=payload)

def get_answer_from_document(message, index_name, namespace):
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


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

if __name__ == '__main__':
    app.run(debug=True)