from dotenv import load_dotenv
import os
import dspy
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tools import search_web
import logging
from db_utils import create_chat_table, save_chat, get_recent_history, clear_history

#configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

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

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}  # top 3 docs
)

#create chat history table
create_chat_table()

#-------------------------------------------------------

#define tools for decisionagent
def search_vector_db(query:str)->str:
    """
    Search the local vector database for relevant information,
    Use this for questions about documents in the knowledge base.
    """
    logging.info(f"Searching vector database for: {query}")

    docs=retriever.invoke(query)
    if docs:
        contents= "\n\n".join([doc.page_content for doc in docs[:3]])
        logging.info(f"Retrieved {len(docs)} documents above score threshold")
        logging.debug(f"Top content: {contents[:200]}...")
        return contents

    logging.warning("No relevant information found in the vector database.")
    return "No relevant information found in the vector database."

# #Takes a user’s question → search for relevant text chunks → return their text content.
# def retrieve(inputs):
#   return [doc.page_content for doc in retriever.invoke(inputs["question"])]

class ToolDecision(dspy.Signature):
    """Based on the Question and coversation History, decide which single
    tool to use and the specific query for that tool."""

    question=dspy.InputField()
    history_context = dspy.InputField(desc="Previous conversation history to follow up the questions.")
    available_tools = dspy.InputField(desc="Available tools: search_vector_db (for internal, domain-specific knowledge), search_web (for current or public information).")

    tool_choice = dspy.OutputField(desc="The name of the tool to be called (search_vector_db or search_web).")
    tool_query = dspy.OutputField(desc="The precise search query string to pass to the chosen tool. Max 10 words.")
    reasoning = dspy.OutputField(desc="Step-by-step thinking to justify the tool choice and query, considering the question and history.")

class FinalAnswer(dspy.Signature):
    """Given the Original Question, the Tool's output, and the Agent's reasoning, generate a final, concise, and helpful answer."""
    question = dspy.InputField()
    tool_output = dspy.InputField(desc="Output/Context retrieved from the selected tool.")
    agent_reasoning = dspy.InputField(desc="The Decision Agent's rationale for the tool used.")
    answer = dspy.OutputField()

#------------------------------------------------------

class DecisionAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decision_module = dspy.ChainOfThought(ToolDecision)
       
    def forward(self, question,history_context):
        response = self.decision_module(
            question=question,
            history_context=history_context,
            available_tools="search_vector_db (for internal knowledge documents), search_web (for current or public information)"
        )
        return response

MAX_ATTEMPTS = 2
FAILURE_MESSAGE = "No relevant information found in the vector database."

class RAGWithDecisionAgent(dspy.Module):
    def __init__(self, tools):
        super().__init__()
        self.decision_agent = DecisionAgent()
        self.final_answer_generator = dspy.Predict(FinalAnswer)
        self.tools = {tool.__name__: tool for tool in tools} # Map tool name string to function object

    def forward(self, question):
        logging.info(f"Processing question with Decision Agent pipeline: {question}")
        
        #get chat history
        attempt = 0
        tool_output = None
        
        # start of the loop
        while attempt < MAX_ATTEMPTS:
            
            # 1. Prepare Context (Conversation History + Dynamic Failure Log)
            history = get_recent_history(limit=3)
            history_context = "Previous conversation:\n" + "\n".join(
                [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history]
            )
            
            # critical : Append failure information to the context for re-planning
            if attempt > 0 and tool_output == FAILURE_MESSAGE:
                failed_tool = decision.tool_choice.strip()
                # Tell the LLM why it's restarting the planning phase
                failure_note = f"\n[CRITICAL NOTE: Attempt {attempt} with tool '{failed_tool}' failed. You MUST select a different available tool.]"
                full_context_for_decision = history_context + failure_note
            else:
                full_context_for_decision = history_context

            # 2. Decision Agent Call (Plan the strategy)
            decision = self.decision_agent(question=question, history_context=full_context_for_decision)
            
            chosen_tool_name = decision.tool_choice.strip()
            chosen_query = decision.tool_query.strip()
            logging.info(f"Attempt {attempt + 1} Plan: Tool: {chosen_tool_name}...")

            # 3. Tool Execution
            if chosen_tool_name in self.tools:
                tool_func = self.tools[chosen_tool_name]
                tool_output = tool_func(chosen_query)
            else:
                tool_output = f"ERROR: Invalid tool chosen: {chosen_tool_name}."

            # 4. Validation Check
            if tool_output != FAILURE_MESSAGE and "ERROR" not in tool_output:
                logging.info("SUCCESS: Relevant information found. Exiting loop.")
                break # Success Generate final answer.

            # 5. Loop Continuation
            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                logging.warning("MAX ATTEMPTS REACHED. Ending conversation.")
        #  end of the loop
        
        # 6. Synthesis (After Loop)
        if tool_output != FAILURE_MESSAGE and "ERROR" not in tool_output:
            response = self.final_answer_generator(
                question=question,
                tool_output=tool_output,
                agent_reasoning=decision.reasoning
            )
            final_answer = response.answer
        else:
            final_answer = "I apologize, I was unable to find relevant information using the available tools after several attempts."
            response = dspy.Prediction(answer=final_answer)

        # 7. Save and Return
        save_chat(question, final_answer)
        response.decision = decision
        return response

if __name__ == "__main__":
    create_chat_table()
    # clear_history()

    rag = RAGWithDecisionAgent(tools=[search_vector_db, search_web])
    
    # testing questions
    questions = [
        "What is  nanobrowser?",  # Should use vector DB
        "Who is the chess champion recently passed away?",  # Should use web search
        "What country he is from?"  # Should use history
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")
        output = rag(question=q)
        print(f"Tool Choice: {output.decision.tool_choice}")
        print(f"Tool Query: {output.decision.tool_query}")
        print(f"Reasoning: {output.decision.reasoning}")
        print(f"Answer: {output.answer}")
        
        # Show the reasoning trace
    dspy.inspect_history(n=1)