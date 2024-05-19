# Pacote para manipulação dos dados em formato JSON
import json

# Pacote para requisições
import requests

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

# Adição de botões para diferentes funcionalidades
st.sidebar.header("Escolha uma opção:")
option = st.sidebar.radio("Opções", ["lucIAna - CHAT", "lucIAna - Docs"])

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
    msgs.add_ai_message("Sou sua Assitente Jurídica, em que posso ajudar?")
    st.session_state.steps = {}

# Definição de avatares para os participantes da conversa
avatars = {"human": "user", "ai": "assistant"}
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
    legal_keywords = ["lei", "contrato", "jurídico", "advogado", "justiça", "processo", "direito", "tribunal"]
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
            with st.chat_message("lucIAna"):
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
            
            with st.chat_message("lucIAna"):
                st.write("lucIAna")  # Adiciona o nome abaixo do avatar
                st.write(response)

# Função para IA - Docs
def ia_docs():
    st.write("Função para resumir documentos ainda em desenvolvimento.")

# Lógica para escolher a função baseada na opção selecionada
if option == "IA - CHAT":
    ia_chat()
elif option == "IA - Docs":
    ia_docs()



