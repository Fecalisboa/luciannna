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
if len(msgs.messages) == 0 ou st.sidebar.button("Reset", key="reset_button"):
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
    st.write("Função para resumir documentos ainda em desenvolvimento.")
    
    # Função para carregar e processar o documento PDF
    def load_doc(list_file_path, chunk_size, chunk_overlap):
        loaders = [PyPDFLoader(x) for x in list_file_path]
        pages = []
        for loader in loaders:
            pages.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_splits = text_splitter.split_documents(pages)
        return doc_splits

    # Função para criar o banco de dados vetorial
    def create_db(splits):
        embeddings = HuggingFaceEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)
        return vectordb

    # Função para inicializar a base de dados vetorial
    def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=st.progress):
        list_file_path = [x.name for x in list_file_obj if x is not None]
        collection_name = create_collection_name(list_file_path[0])
        doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
        vector_db = create_db(doc_splits)
        return vector_db, collection_name, "Complete!"

    # Função para criar o nome da coleção
    def create_collection_name(filepath):
        collection_name = Path(filepath).stem
        collection_name = collection_name.replace(" ", "-")
        collection_name = unidecode(collection_name)
        collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
        collection_name = collection_name[:50]
        if len(collection_name) < 3:
            collection_name = collection_name + 'xyz'
        if not collection_name[0].isalnum():
            collection_name = 'A' + collection_name[1:]
        if not collection_name[-1].isalnum():
            collection_name = collection_name[:-1] + 'Z'
        print('Filepath: ', filepath)
        print('Collection name: ', collection_name)
        return collection_name

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

    # Inicialização dos elementos de interface para upload e processamento de documentos PDF
    uploaded_files = st.file_uploader("Upload your PDF documents (single or multiple)", type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk size", 100, 1000, 600, 20)
    chunk_overlap = st.slider("Chunk overlap", 10, 200, 40, 10)

    if st.button("Generate vector database"):
        if uploaded_files:
            vector_db, collection_name, status = initialize_database(uploaded_files, chunk_size, chunk_overlap)
            st.success("Vector database created successfully!")

            # Inicialização do LLM chain
            llm_option = 1  # Índice para Mistral v0.3
            llm_temperature = 0.7
            max_tokens = 1024
            top_k = 3

            qa_chain = initialize_llmchain("mistralai/Mistral-7B-Instruct-v0.3", llm_temperature, max_tokens, top_k, vector_db)
            st.success("LLM chain initialized successfully!")

# Lógica para escolher a função baseada na opção selecionada
if option == "IA - CHAT":
    ia_chat()
elif option == "IA - Docs":
    ia_docs()

