# Pacote para manipulação dos dados em formato JSON
import json

# Framework para criação de aplicações web
import streamlit as st  

# Para criação e execução de agentes conversacionais
from langchain.agents import ConversationalChatAgent, AgentExecutor  

# Callback para interação com a interface do Streamlit
from langchain_community.callbacks import StreamlitCallbackHandler  

# Memória para armazenar o histórico de conversa
from langchain.memory import ConversationBufferMemory  

# Histórico de mensagens para o Streamlit
from langchain_community.chat_message_histories import StreamlitChatMessageHistory 

# Integração com o modelo de linguagem da Cohere
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ferramenta de busca DuckDuckGo para o agente 
from langchain_community.tools import DuckDuckGoSearchRun  

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Pacotes para processamento de documentos PDF e criação de banco de dados vetorial
import os
from pathlib import Path
import re
from unidecode import unidecode
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint

# Configuração do título da página
st.set_page_config(page_title="lucIAna")

# Criação de colunas para layout da página
# Define a proporção das colunas
col1, col4 = st.columns([4, 1])  

# Configuração da primeira coluna para exibir o título do projeto
with col1:
    st.title("lucIAna")

# Definição da chave de API da Cohere
cohere_api_key = "OGY2ZCgZ4351TM0pXzRNeJLpw6o9GhyfWA3r05eW"

# Obtenha o token da variável de ambiente
hf_api_key = "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"

# Adição de botões para diferentes funcionalidades
st.sidebar.header("Escolha uma opção:")
option = st.sidebar.radio("Opções", ["IA - CHAT", "IA - Docs"])

# Função para carregar dados do arquivo JSON
def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Função para salvar dados no arquivo JSON
def save_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Caminho do arquivo JSON
json_file_path = 'chat_history.json'

# Carregar histórico de mensagens do arquivo JSON
chat_history = load_data(json_file_path)

# Inicialização do histórico de mensagens no Streamlit
if "msgs" not in st.session_state:
    st.session_state.msgs = StreamlitChatMessageHistory()

msgs = st.session_state.msgs

# Configuração da memória do chat
memory = ConversationBufferMemory(chat_memory=msgs, 
                                  return_messages=True, 
                                  memory_key="chat_history", 
                                  output_key="output")

# Verificação para limpar o histórico de mensagens ou iniciar a conversa
if len(msgs.messages) == 0 or st.sidebar.button("Reset", key="reset_button"):
    msgs.clear()
    msgs.add_ai_message("Sou sua Assistente Jurídica, em que posso ajudar?")
    st.session_state.steps = {}

# Definição de avatares para os participantes da conversa
avatars = {"human": "user", "ai": "👩‍🎤"}
names = {"human": "Você", "ai": "lucIAna"}

# Itera sobre cada mensagem no histórico de mensagens
for idx, msg in enumerate(msgs.messages):  
    # Cria uma mensagem no chat com o avatar correspondente ao tipo de usuário (humano ou IA)
    with st.chat_message(avatars[msg.type]):  
        st.write(names[msg.type])  # Adiciona o nome abaixo do avatar

        # Itera sobre os passos armazenados para cada mensagem, se houver
        for step in st.session_state.steps.get(str(idx), []):  
            # Se o passo atual indica uma exceção, pula para o próximo passo
            if step[0].tool == "_Exception":  
                continue

            # Cria um expander para cada ferramenta usada na resposta, mostrando o input
            with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"): 
                # Exibe o log de execução da ferramenta 
                st.write(step[0].log)  
                # Exibe o resultado da execução da ferramenta
                st.write(f"**{step[1]}**")  

        # Exibe o conteúdo da mensagem no chat
        st.write(msg.content)  

# Função para verificar se a pergunta é jurídica
def is_legal_question(question):
    legal_keywords = ["lei", "contrato", "jurídico", "advogado", "justiça", "processo", "direito", "tribunal", "artigo", "bom dia", "boa tarde", "boa noite", "oi", "olá"]
    return any(keyword in question.lower() for keyword in legal_keywords)

# Função para o chat da IA
def ia_chat():
    # Campo de entrada para novas mensagens do usuário
    if prompt := st.chat_input(placeholder="Digite uma pergunta para começar!", key="chat_input"):
        st.chat_message("user").write(prompt)
        
        if is_legal_question(prompt):
            # Configuração do modelo de linguagem da Cohere
            llm = ChatCohere(cohere_api_key=cohere_api_key)
            
            # Configuração da ferramenta de busca do agente
            mecanismo_busca = [DuckDuckGoSearchRun(name="Search")]
            
            # Criação do agente conversacional com a ferramenta de busca
            chat_dsa_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=mecanismo_busca)
            
            # Executor para o agente, incluindo memória e tratamento de erros
            executor = AgentExecutor.from_agent_and_tools(agent=chat_dsa_agent,
                                                          tools=mecanismo_busca,
                                                          memory=memory,
                                                          return_intermediate_steps=True,
                                                          handle_parsing_errors=True)

            # Definição do prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Você é um advogado experiente e deve fornecer respostas com base em seu conhecimento jurídico."),
                ("user", "{input}")
            ])
            
            # Criação do chain
            chain = prompt_template | llm | StrOutputParser()

            # Execução do chain com a entrada do usuário
            response = chain.invoke({"input": prompt})

            # Adicionar a resposta da IA ao histórico de mensagens
            msgs.add_ai_message(response)

            # Salvar a pergunta e resposta no arquivo JSON
            chat_history.append({"role": "user", "content": prompt})
            chat_history.append({"role": "ai", "content": response})
            save_data(json_file_path, chat_history)

            # Exibir a resposta do assistente
            with st.chat_message("🤖"):
                st.write("lucIAna")  # Adiciona o nome abaixo do avatar
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)  
                response = executor(prompt, callbacks=[st_cb])
                st.write(response["output"])
                # Armazenamento dos passos intermediários
                st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]  
        else:
            # Resposta para perguntas não jurídicas
            response = "Desculpe, fui treinada apenas para responder perguntas sobre temas jurídicos."
            msgs.add_ai_message(response)
            chat_history.append({"role": "user", "content": prompt})
            chat_history.append({"role": "ai", "content": response})
            save_data(json_file_path, chat_history)
            
            with st.chat_message("🤖"):
                st.write("lucIAna")  # Adiciona o nome abaixo do avatar
                st.write(response)

# Função para IA - Docs
def ia_docs():
    st.write("Carregue e processe seus documentos PDF.")
    
    # Função para carregar e processar o documento PDF
    def load_doc(list_file_path):
        # Processing for one document only
        # loader = PyPDFLoader(file_path)
        # pages = loader.load()
        loaders = [PyPDFLoader(x) for x in list_file_path]
        pages = []
        for loader in loaders:
            pages.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1024, 
            chunk_overlap = 64 
        )  
        doc_splits = text_splitter.split_documents(pages)
        return doc_splits

    # Função para criar o banco de dados vetorial
    def create_db(splits):
        embeddings = HuggingFaceEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)
        return vectordb

    # Função para inicializar a base de dados vetorial
    def initialize_database(list_file_obj):
        doc_splits = load_doc(list_file_obj, chunk_size=600, chunk_overlap=40)
        vector_db = create_db(doc_splits)
        return vector_db

    # Função para inicializar o LLM chain usando Mistral v0.3 com a API da Hugging Face
    def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=st.progress):
        progress(0.1, "Initializing HF tokenizer...")
        progress(0.5, "Initializing HF Hub...")

        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            huggingfacehub_api_token=hf_api_key,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )

        progress(0.75, "Defining buffer memory...")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key='answer',
            return_messages=True
        )
        retriever = vector_db.as_retriever()
        progress(0.8, "Defining retrieval chain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            chain_type="stuff", 
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        progress(0.9, "Done!")
        return qa_chain

    # Initialize database
def initialize_database(list_file_obj, progress=gr.Progress()):
    # Create a list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # Load document and create splits
    doc_splits = load_doc(list_file_path)
    # Create or load vector database
    vector_db = create_db(doc_splits)
    return vector_db, "Database created!"

# Initialize LLM
def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "QA chain initialized. Chatbot is ready!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    # Generate response using QA chain
    response = qa_chain.invoke({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page
    

def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    return list_file_path


def demo():
    with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        gr.HTML("<center><h1>RAG PDF chatbot</h1><center>")
        gr.Markdown("""<b>Query your PDF documents!</b> This AI agent is designed to perform retrieval augmented generation (RAG) on PDF documents. The app is hosted on Hugging Face Hub for the sole purpose of demonstration. \
        <b>Please do not upload confidential documents.</b>
        """)
        with gr.Row():
            with gr.Column(scale = 86):
                gr.Markdown("<b>Step 1 - Upload PDF documents and Initialize RAG pipeline</b>")
                with gr.Row():
                    document = gr.Files(height=300, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF documents")
                with gr.Row():
                    db_btn = gr.Button("Create vector database")
                with gr.Row():
                        db_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Vector database status", 
                gr.Markdown("<style>body { font-size: 16px; }</style><b>Select Large Language Model (LLM) and input parameters</b>")
                with gr.Row():
                    llm_btn = gr.Radio(list_llm_simple, label="Available LLMs", value = list_llm_simple[0], type="index") # info="Select LLM", show_label=False
                with gr.Row():
                    with gr.Accordion("LLM input parameters", open=False):
                        with gr.Row():
                            slider_temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.5, step=0.1, label="Temperature", info="Controls randomness in token generation", interactive=True)
                        with gr.Row():
                            slider_maxtokens = gr.Slider(minimum = 128, maximum = 9192, value=4096, step=128, label="Max New Tokens", info="Maximum number of tokens to be generated",interactive=True)
                        with gr.Row():
                                slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k", info="Number of tokens to select the next token from", interactive=True)
                with gr.Row():
                    qachain_btn = gr.Button("Initialize Question Answering Chatbot")
                with gr.Row():
                        llm_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Chatbot status", 

            with gr.Column(scale = 200):
                gr.Markdown("<b>Step 2 - Chat with your Document</b>")
                chatbot = gr.Chatbot(height=505)
                with gr.Accordion("Relevent context from the source document", open=False):
                    with gr.Row():
                        doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                        source1_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                        source2_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                        source3_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a question", container=True)
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.ClearButton([msg, chatbot], value="Clear")
            
        # Preprocessing events
        db_btn.click(initialize_database, \
            inputs=[document], \
            outputs=[vector_db, db_progress])
        qachain_btn.click(initialize_LLM, \
            inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], \
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)

        # Chatbot events
        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        clear_btn.click(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
    demo.queue().launch(debug=True)



# Lógica para escolher a função baseada na opção selecionada
if option == "IA - CHAT":
    ia_chat()
elif option == "IA - Docs":
    ia_docs()

if __name__ == "__main__":
    demo()
