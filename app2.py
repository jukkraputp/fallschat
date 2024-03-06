import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from translate import Translator

st.set_page_config(page_title='DHV AI Startup', layout='wide')
st.title("DHV AI Startup Chatbot Demo")
st.title("ประเมินความเสี่ยงต่อการพลัดตกหกล้ม")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if prompt:
        # Google Translate
        try:
            translator = Translator(from_lang='th', to_lang='en')
            translated_text = translator.translate(prompt)
        except Exception as e:
            st.write(f"Error translating: {str(e)}")
            st.stop()

        # Prepare the DB.
        api_key = st.secrets["OPENAI_API_KEY"]
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(translated_text, k=3)
        if not results or (results and results[0][1] < 0.7):
            st.write("ไม่สามารถค้นหาคำตอบขณะนี้ได้ โปรดติดต่อแพทย์ของท่าน")
            st.stop()

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=translated_text)
        st.write(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"<span style='color:red'>{response_text}</span>\nSources: {sources}"
        st.write(formatted_response, unsafe_allow_html=True)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            if st.session_state.messages:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                )
                assistant_response = response.choices[0].message.content
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                # Check if assistant response indicates high risk
                if "high risk" in assistant_response:
                    st.write("Please go to [this link](https://www.hopkinsmedicine.org/institute_nursing/_docs/JHFRAT/JHFRAT%20Tools/JHFRAT_acute%20care%20original_6_22_17.pdf) for the fall risk assessment.")
                else:
                    st.write("Please go to [this link](https://www.samitivejhospitals.com/article/detail/fall-risk-assessment) for the fall risk assessment.")
            else:
                st.markdown("Please enter a message.")