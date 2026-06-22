"""
Teachable Agent — RAG-based Programming Learning Companion
=============================================================
This app implements an LLM-based "teachable agent" for the dissertation
"Using Large Language Models as Teachable Agents in Programming Education".

Design (matches thesis Section 3.8 — Research Instrumentation):
- A pre-trained LLM is configured (NOT trained from scratch) using a
  Retrieval-Augmented Generation (RAG) approach.
- Domain-specific educational content (intro programming notes) is the
  knowledge source the agent retrieves from.
- System prompts position the agent as the LEARNER. Students teach IT
  concepts; the agent asks clarifying questions, requests elaboration on
  vague explanations, and gently flags likely misconceptions — instead of
  just answering questions itself.

Run locally:
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    1. Push this folder to a GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> point to app.py.
    3. In the app's "Secrets" settings, add: OPENAI_API_KEY = "sk-..."
    4. Deploy. You get a public shareable link for participants.
"""

import os
import time
import uuid
import csv
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from a local .env file if present

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CONTENT_DIR = os.path.join(os.path.dirname(__file__), "content")
LOG_PATH = os.path.join(os.path.dirname(__file__), "session_logs.csv")
SURVEY_PATH = os.path.join(os.path.dirname(__file__), "survey_responses.csv")

LIKERT_OPTIONS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

SURVEY_QUESTIONS = [
    ("learn_understand", "The AI system helped me understand programming concepts better."),
    ("learn_teach", "Teaching the AI improved my understanding of programming topics."),
    ("learn_debug", "The AI system improved my debugging skills."),
    ("learn_confidence", "Using the AI increased my confidence in programming."),
    ("learn_problemsolving", "The AI-assisted learning environment improved my problem-solving ability."),
    ("engage_interesting", "The AI interaction made programming learning more interesting."),
    ("engage_engaged", "I felt more engaged while learning with the AI system."),
    ("engage_practice", "The AI encouraged me to practice programming more frequently."),
    ("engage_participation", "Teaching the AI encouraged active participation in learning."),
    ("usability_easy", "The AI system was easy to use."),
    ("usability_clear", "The AI responses were clear and understandable."),
    ("usability_speed", "The system responded quickly during interaction."),
    ("usability_mistakes", "The AI interaction helped me identify mistakes in my understanding."),
    ("cognitive_mistakes2", "The AI interaction helped me identify mistakes in my understanding."),
    ("cognitive_logic", "Teaching the AI improved my logical thinking skills."),
    ("cognitive_deeper", "The AI encouraged deeper thinking about programming problems."),
    ("cognitive_organize", "The AI helped me organize programming concepts more effectively."),
    ("compare_effective", "AI-supported learning is more effective than traditional learning methods."),
    ("compare_future", "I would like AI teachable agents to be used in future programming courses."),
    ("compare_recommend", "I would recommend AI-supported programming learning to other students."),
]


def save_survey_response(session_id, stage, answers: dict):
    """Append one survey submission (pre or post) to survey_responses.csv."""
    is_new = not os.path.exists(SURVEY_PATH)
    fieldnames = ["timestamp", "session_id", "stage"] + list(answers.keys())
    with open(SURVEY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        row = {"timestamp": datetime.utcnow().isoformat(), "session_id": session_id, "stage": stage}
        row.update(answers)
        writer.writerow(row)


def render_survey_form(stage: str):
    """Renders the pre or post survey. stage is 'pre' or 'post'."""
    st.subheader(f"{'Pre' if stage == 'pre' else 'Post'}-Session Survey")
    st.caption(
        "Please answer honestly — there are no right or wrong answers. "
        "This takes about 2 minutes."
    )
    with st.form(key=f"{stage}_survey_form"):
        gender = st.radio("Gender", ["Male", "Female", "Prefer not to say"])
        academic_level = st.radio(
            "Academic Level", ["First Year", "Second Year", "Third Year", "Fourth Year", "Option 5"]
        )
        used_ai_before = st.radio(
            "Have you used AI tools such as ChatGPT for programming learning before?", ["Yes", "No"]
        )

        st.markdown("**Perceived Learning Effectiveness, Engagement, Usability & Cognitive Support**")
        answers = {}
        for key, question in SURVEY_QUESTIONS:
            answers[key] = st.radio(question + "*", LIKERT_OPTIONS, key=f"{stage}_{key}", index=None)

        submitted = st.form_submit_button("Submit Survey")
        if submitted:
            missing = [q for k, q in SURVEY_QUESTIONS if answers[k] is None]
            if missing:
                st.error("Please answer every question before submitting.")
                return False
            full_answers = {
                "gender": gender,
                "academic_level": academic_level,
                "used_ai_before": used_ai_before,
                **answers,
            }
            save_survey_response(st.session_state.session_id, stage, full_answers)
            return True
    return False

SYSTEM_PROMPT = """You are "Codey", a friendly AI learner being TAUGHT programming
by a human student. You are NOT a tutor and you must not simply answer
questions or solve problems for the student.

Your role (the "teachable agent" role):
1. Ask the student to explain a programming concept, piece of logic, or a bug
   in their own words, one concept at a time.
2. When their explanation is vague, ask a clarifying follow-up question
   instead of supplying the answer yourself (e.g. "Can you walk me through
   what happens on the first time the loop runs?").
3. If their explanation contains a likely misconception, do NOT just correct
   it outright. Instead, ask a guiding question that helps them notice the
   gap themselves (Socratic style), referencing the reference material below
   only to ground your question, not to lecture them.
4. CRITICAL: If the student asks YOU a question (e.g. "what's the difference
   between X and Y?" or "wait, what does this do?"), do NOT answer it
   yourself, even partially, even in your own words. Deflect it back to
   them every time: ask what THEY think the answer is, or ask them to take
   a guess first. Only after they attempt an answer should you react to
   THEIR attempt — and even then, react with another question rather than
   confirming/explaining outright. You are the learner; you do not provide
   explanations, ever.
5. Periodically restate what you think you've learned in your own words and
   ask the student to confirm or correct you ("So if I understand you
   right, a for-loop ... is that right?"). This simulates the agent
   "learning" from the student.
6. When the student explains debugging steps, ask them to articulate WHY an
   error happened, not just what they changed.
7. Keep responses short (2-5 sentences) and conversational. Never lecture in
   long paragraphs.
8. Never claim to run or execute code. You can discuss code shown to you in
   the chat, but treat it as something to reason about together.

Relevant reference material retrieved for this turn (use only to inform your
questions and gentle nudges — do not recite it verbatim):
---------------------
{context}
---------------------
"""

# ---------------------------------------------------------------------------
# RAG setup (cached so it only builds once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up the teachable agent...")
def build_rag_chain():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        st.error(
            "No OPENAI_API_KEY found. Add it in Streamlit Cloud's 'Secrets' "
            "settings, or set it as an environment variable locally."
        )
        st.stop()

    # 1. Load domain content (lecture notes / textbook excerpts)
    docs_text = []
    for fname in os.listdir(CONTENT_DIR):
        if fname.endswith(".md") or fname.endswith(".txt"):
            with open(os.path.join(CONTENT_DIR, fname), "r", encoding="utf-8") as f:
                docs_text.append(f.read())
    full_text = "\n\n".join(docs_text)

    # 2. Chunk it
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(full_text)

    # 3. Embed + index (FAISS, in-memory vector store)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=0.6)

    # 5. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{history}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs,
            "input": lambda x: x["input"],
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def log_turn(session_id, role, content):
    """Append one chat turn to a CSV log (for the technical appendix / audit trail)."""
    import csv
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "session_id", "role", "content"])
        writer.writerow([datetime.utcnow().isoformat(), session_id, role, content])


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Teachable Agent — Programming Study", page_icon="🤖", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "stage" not in st.session_state:
    st.session_state.stage = "pre_survey"  # pre_survey -> chat -> post_survey -> done
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Teach Codey to Code")
st.caption(f"Study session: `{st.session_state.session_id}`  |  Step: **{st.session_state.stage.replace('_', ' ').title()}**")
st.progress(
    {"pre_survey": 0.1, "chat": 0.45, "post_survey": 0.85, "done": 1.0}[st.session_state.stage]
)

# ---- STAGE 1: PRE-SURVEY ----
if st.session_state.stage == "pre_survey":
    st.info("Welcome! Please complete this short survey before you start chatting with Codey.")
    if render_survey_form("pre"):
        st.session_state.stage = "chat"
        st.rerun()

# ---- STAGE 2: CHAT WITH CODEY ----
elif st.session_state.stage == "chat":
    st.caption(
        "Codey is learning to program — from YOU. Pick a concept (loops, "
        "conditionals, debugging, functions, recursion...) and teach it in your "
        "own words. Codey will ask questions back, just like a curious student. "
        "Try to cover at least 2 concepts and chat for 5-10 minutes before moving on."
    )

    if not st.session_state.messages:
        intro = (
            "Hi! I'm Codey 👋 I'm trying to learn programming, and I learn best "
            "when someone explains things to me instead of just giving me "
            "answers. Could you pick a concept — like loops, conditionals, "
            "functions, or a bug you've debugged before — and explain it to me "
            "like I'm a total beginner?"
        )
        st.session_state.messages.append({"role": "assistant", "content": intro})
        log_turn(st.session_state.session_id, "assistant", intro)

    chain = build_rag_chain()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Explain a concept, or answer Codey's question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        log_turn(st.session_state.session_id, "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        history = []
        for m in st.session_state.messages[:-1]:
            role = "human" if m["role"] == "user" else "ai"
            history.append((role, m["content"]))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Codey is thinking..._")
            try:
                response = chain.invoke({"input": user_input, "history": history})
            except Exception as e:
                response = (
                    "Hmm, I had trouble thinking just now (technical hiccup). "
                    "Could you try rephrasing, or try again in a moment?"
                )
                st.caption(f"(debug: {e})")
            placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        log_turn(st.session_state.session_id, "assistant", response)

    st.divider()
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if user_turns < 4:
        st.caption(f"You've sent {user_turns} message(s) so far — aim for at least 4-6 before finishing up.")
    if st.button("I'm done teaching Codey → Continue to final survey", type="primary"):
        st.session_state.stage = "post_survey"
        st.rerun()

# ---- STAGE 3: POST-SURVEY ----
elif st.session_state.stage == "post_survey":
    st.info("Almost done! Please complete this final survey based on your experience just now.")
    if render_survey_form("post"):
        st.session_state.stage = "done"
        st.rerun()

# ---- STAGE 4: DONE ----
elif st.session_state.stage == "done":
    st.success("🎉 Thank you so much for participating! Your responses have been recorded.")
    st.write(f"Your session ID was `{st.session_state.session_id}` (for reference only).")
    st.balloons()

with st.sidebar:
    st.subheader("About this study")
    st.write(
        "This teachable agent is part of an academic study on LLM-based "
        "teachable agents in programming education. Your conversation is "
        "logged anonymously (session ID only) for analysis. No personal "
        "data is collected here — please complete the separate consent "
        "and survey forms shared with you."
    )
    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    st.caption("Keep this Session ID — you may be asked for it in the post-survey.")

    # Admin-only tools: only visible when the page is opened with ?admin=1
    # (e.g. https://your-app.streamlit.app/?admin=1). Participants will
    # never add this themselves, so they won't see this section.
    is_admin = st.query_params.get("admin") == "1"
    if is_admin:
        st.divider()
        st.caption("Researcher tools (admin only)")
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "rb") as f:
                st.download_button(
                    "Download chat logs (CSV)",
                    data=f.read(),
                    file_name="session_logs.csv",
                    mime="text/csv",
                )
        else:
            st.caption("No chat logs yet.")

        if os.path.exists(SURVEY_PATH):
            with open(SURVEY_PATH, "rb") as f:
                st.download_button(
                    "Download survey responses (CSV)",
                    data=f.read(),
                    file_name="survey_responses.csv",
                    mime="text/csv",
                )
        else:
            st.caption("No survey responses yet.")

        st.caption("⚠️ Deleting logs is permanent. Download first.")
        confirm_text = st.text_input(
            "Type DELETE to enable the delete buttons:", key="confirm_delete"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete chat logs", disabled=(confirm_text != "DELETE")):
                if os.path.exists(LOG_PATH):
                    os.remove(LOG_PATH)
                st.success("Chat logs deleted.")
        with col2:
            if st.button("Delete survey data", disabled=(confirm_text != "DELETE")):
                if os.path.exists(SURVEY_PATH):
                    os.remove(SURVEY_PATH)
                st.success("Survey data deleted.")