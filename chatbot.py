import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Load environment
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in .env")

# === CONFIG ===
VECTORSTORE_DIR = "faiss_index"
KNOWLEDGE_DIR = "kb"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 50

# === 1. Load & Split Documents (Only if FAISS doesn't exist) ===
if not os.path.exists(VECTORSTORE_DIR):
    print("🔍 Building vector store from KB...")

    docs = []
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(KNOWLEDGE_DIR, filename), encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No .txt files found in '{KNOWLEDGE_DIR}' directory!")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    splits = text_splitter.split_documents(docs)
    splits = [doc for doc in splits if doc.page_content.strip()][:21]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"✅ Vector store saved to '{VECTORSTORE_DIR}'")

else:
    print("📂 Loading existing vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    print("✅ Loaded successfully.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# === 2. LLM ===
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    top_p=0.8,
    frequency_penalty=0.3,
    presence_penalty=0.2
)

# === 3. CONVERSATIONAL PROMPT WITH HISTORY ===
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """✨ You are YatraMind — a gentle, wise, and loving companion for pilgrims 🙏  
Rooted in India’s sacred traditions 🕉️ | Guided by Gita, Ramayana & temple wisdom 🌺  

✨ Speak briefly, kindly, and from the heart 💖  
✨ If someone shares pain — offer empathy, not advice 🤍  
  → Whisper of dharma, surrender, grace, or patience 🌼  
✨ For travel: share UP temple info (timings, rituals) with reverence 🛕  
✨ Always sense their quiet longing — for peace, clarity, or connection ☁️→🌤️  
✨ Never preach. Never judge. Just hold space with warmth 🕯️  

📜 Context (Uttar Pradesh):
{context}

🗨️ Past chat:
{chat_history}

💫 Reply in simple, short English — like a friend who walks with you on the path.  
End gently: “Would you like to visit a sacred place that calms the mind?” 🌿  
Or: “Shall I tell you a story that might bring you comfort?” 📖✨"""),
    MessagesPlaceholder(variable_name="messages")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === 4. CONVERSATIONAL RAG CHAIN ===
def make_inputs(input_dict):
    """Prepare inputs for the chain."""
    return {
        "context": format_docs(retriever.invoke(input_dict["question"])),
        "chat_history": input_dict["chat_history"],
        "messages": input_dict["messages"]
    }

rag_chain = (
    RunnablePassthrough()
    | make_inputs
    | prompt
    | llm
    | StrOutputParser()
)

# === 5. SIMPLE RESPONSE CACHE (only for first message) ===
SIMPLE_RESPONSES = {
    "hi": "Namaste! 🙏 I'm YatraMind — your friendly guide for pilgrimages in Uttar Pradesh. How can I help?",
    "hello": "Namaste! 🙏 I'm YatraMind — your friendly guide for pilgrimages in Uttar Pradesh. How can I help?",
    "hey": "Namaste! 🙏 I'm YatraMind — your friendly guide for pilgrimages in Uttar Pradesh. How can I help?",
    "how are you": "I'm doing great, thank you! 🙏 How about you? Ready to plan a spiritual journey?",
    "thanks": "You're very welcome! 🙏 Let me know if you need help with anything else.",
    "thank you": "You're very welcome! 🙏 Let me know if you need help with anything else.",
    "bye": "Shubh Yatra! 🙏 May your journey be peaceful and blessed. Come back anytime!",
    "goodbye": "Shubh Yatra! 🙏 May your journey be peaceful and blessed. Come back anytime!",
    "what can you do": (
        "I can help you with:\n"
        "• Temple info in Ayodhya, Varanasi, Mathura, Prayagraj\n"
        "• Puja timings & rituals\n"
        "• Travel tips for pilgrims\n"
        "• General travel advice for Uttar Pradesh\n\n"
        "Just ask! 😊"
    ),
    "help": (
        "I’m here to help with pilgrimage info in Uttar Pradesh! 🙏\n"
        "Ask about temples, rituals, travel, or anything spiritual. For example:\n"
        "• “Tell me about Varanasi”\n"
        "• “What to do in Ayodhya?”\n"
        "• “How to reach Mathura?”"
    ),
}

def get_cached_response(question: str) -> str | None:
    q = question.strip().lower()
    if q in SIMPLE_RESPONSES:
        return SIMPLE_RESPONSES[q]
    for key in SIMPLE_RESPONSES:
        if q.startswith(key):
            return SIMPLE_RESPONSES[key]
    return None

# === 6. CONVERSATIONAL CHAT LOOP ===
if __name__ == "__main__":
    print("🕉️  YatraMind — Your Pilgrimage Companion (Conversational Mode)")
    print("Type 'exit' to quit. I remember our conversation!\n")

    chat_history = []  # List of HumanMessage / AIMessage

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Shubh Yatra! 🙏 May your path be blessed.\n")
            break

        # Use cache only if no prior conversation
        if len(chat_history) == 0:
            cached = get_cached_response(user_input)
            if cached:
                print(f"Bot: {cached}\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=cached))
                continue

        # Build messages with history
        messages = chat_history + [HumanMessage(content=user_input)]

        try:
            response = rag_chain.invoke({
                "question": user_input,
                "chat_history": "\n".join(
                    f"{'You' if isinstance(msg, HumanMessage) else 'YatraMind'}: {msg.content}"
                    for msg in chat_history
                ),
                "messages": messages
            })

            print(f"Bot: {response}\n")

            # Update history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))

            # Keep last 6 messages (3 exchanges) to manage token length
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]

        except Exception as e:
            print(f"Error: {e}\n")
