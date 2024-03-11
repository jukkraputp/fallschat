import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from translate import Translator

st.set_page_config(page_title="DHV AI Startup", layout="wide")
st.title("DHV AI Startup Chatbot Demo")
st.title("ประเมินความเสี่ยงต่อการพลัดตกหกล้ม")

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
    # st.session_state.messages.append(
    #     {
    #         "role": "system",
    #         "content": "Please give us the falls risk at the first word of your answer, either 'HIGH' or 'LOW'.",
    #     },
    # )

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
            translator = Translator(from_lang="th", to_lang="en")
            translated_text = (
                translator.translate(prompt)
                + """
                \n What is the fall risk score? (list each score and explain why it is)
                \n Is the risk is 'Low', 'Moderate' or 'High'?
                """
            )
        except Exception as e:
            st.write(f"Error translating: {str(e)}")
            st.stop()

        # Prepare the DB.
        embedding_function = OpenAIEmbeddings()
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embedding_function
        )

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(translated_text, k=3)
        if not results or (results and results[0][1] < 0.7):
            st.write("ไม่สามารถค้นหาคำตอบขณะนี้ได้ โปรดติดต่อแพทย์ของท่าน")
            st.stop()

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=translated_text)
        st.write(prompt)

        response_text = ChatOpenAI().invoke(prompt)

        formatted_response = f"""
        <span style='color:red'>
        {response_text.content}
        </span>
        """
        st.write(formatted_response, unsafe_allow_html=True)

        # Display assistant response in chat message container
        # Check if assistant response indicates high risk
        if "HIGH" in response_text.content.upper() or "MODERATE" in response_text.content.upper():
            st.write(
                """
                Please go to [this link](3.106.58.71:8501) for calculate risk from x-ray image.\n
                Please go to [this link](https://www.hopkinsmedicine.org/institute_nursing/_docs/JHFRAT/JHFRAT%20Tools/JHFRAT_acute%20care%20original_6_22_17.pdf) for the fall risk assessment.
                """
            )
        else:
            st.write(
                "Please go to [this link](https://www.samitivejhospitals.com/article/detail/fall-risk-assessment) for the fall risk assessment."
            )
