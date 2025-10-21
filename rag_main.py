from dotenv import load_dotenv
import os
import dspy
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tools import search_web

# Load variables from .env into environment
load_dotenv()

# Get the API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

#configure the llm
lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_api_key)
dspy.configure(lm=lm)

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load the persisted vector store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

#query the vector store
question = "What is nanobrowser"
retriever = db.as_retriever()

#for storing past chat history
chat_history=[]

#define tools for ReAct 
def search_vector_db(query:str)->str:
    """
    Search the local vector database for relevant information,
    Use this for questionsabout documents in the knowledge base.
    """
    docs=retriever.invoke(query)
    if docs:
        return "\n\n".join([doc.page_content for doc in docs[:3]])
    return "No relevant information found in the vector database."

# #Takes a user’s question → search for relevant text chunks → return their text content.
# def retrieve(inputs):
#   return [doc.page_content for doc in retriever.invoke(inputs["question"])]

class RAGWithReAct(dspy.Module):
    def __init__(self):

        #ReAct module that can use tools
        self.react = dspy.ReAct(
            signature="question->answer",
            tools=[search_vector_db, search_web],
            max_iters=1 #max reasoning iterations
        )
        
    def forward(self, question):
        # Build context from chat history
        history_context = ""
        if chat_history:
            history_context = "Previous conversation:\n"
            for entry in chat_history[-3:]:  # Last 3 exchanges
                history_context += f"Q: {entry['question']}\nA: {entry['response'].answer}\n"
        
        # Enhance question with history if available
        enhanced_question = question
        if history_context:
            enhanced_question = f"{history_context}\nCurrent question: {question}"
        
        self.react.set_lm(lm=dspy.LM("openai/gpt-4.1", temperature=0.7))
        # Let ReAct decide which tools to use
        response = self.react(question=enhanced_question)
        
        # Store in chat history
        chat_history.append({
            "question": question,
            "response": response
        })
        
        return response

if __name__ == "__main__":
    rag = RAGWithReAct()
    
    # testing questions
    questions = [
        "What is nanobrowser?",  # Should use vector DB
        "What are the latest AI trends in 2025?",  # Should use web search
        "Can you elaborate on the previous answer?"  # Should use history
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")
        output = rag(question=q)
        print(f"Answer: {output.answer}")
        
        # Show the reasoning trace
        if hasattr(output, 'trajectories'):
            print(f"\nReasoning Trace:")
            for i, step in enumerate(output.trajectories, 1):
                print(f"  Step {i}: {step}")
