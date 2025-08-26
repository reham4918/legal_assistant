# C:/Users/LENOVO/PycharmProjects/legal_assistant/app.py

# -*- coding: utf-8 -*-
"""
مساعد قانوني ذكي بواجهة ويب باستخدام Flask (النسخة الحوارية)
"""
from flask import Flask, render_template, request, session, jsonify
from assistant_core import LegalAssistant
import os
import traceback
from dotenv import load_dotenv

# --- تحميل متغيرات البيئة من ملف .env ---
load_dotenv()

# --- استيراد مكونات LangChain ---
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool

# --- تهيئة تطبيق Flask ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- تحميل وتهيئة المساعد القانوني ---
legal_assistant = LegalAssistant()

# --- بناء الأداة بشكل صريح هنا ---
tools = [
    Tool(
        name="analyze_case",
        func=legal_assistant.analyze_case,
        description="""
        استخدم هذه الأداة فقط عندما يكون لديك وصف كامل ومفصل للحالة من المستخدم.
        تأخذ هذه الأداة استعلامًا مفصلاً من المستخدم حول قضية في قانون العمل، وتسترجع المواد القانونية ذات الصلة،
        وتقدم تحليلاً قانونياً كاملاً. لا تستخدم هذه الأداة إذا كان الاستعلام قصيرًا أو غامضًا أو يفتقر
        إلى تفاصيل حاسمة مثل مدة الخدمة، أو سبب الفصل، أو ما إذا كان قد تم توجيه إنذار.
        إذا كانت التفاصيل ناقصة، اطرح على المستخدم أسئلة توضيحية أولاً.
        """
    )
]

# --- بناء "عقل" الوكيل (Agent) ---

# 1. تهيئة النموذج اللغوي
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("لم يتم العثور على مفتاح GROQ_API_KEY. يرجى التأكد من وجود ملف .env في المجلد الرئيسي للمشروع وأن المتغير GROQ_API_KEY معرف بداخله.")

llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-70b-8192",
    api_key=groq_api_key
)

# 2. بناء قالب الأوامر الخاص بالوكيل (تم تحسينه بشكل كبير)
agent_prompt_template = """
You are a helpful and expert Qatari labor law assistant. Your primary language for conversation is Arabic.

You have access to the following tools:
{tools}

To answer the user's question, follow this reasoning process:

1.  **Analyze the user's query and conversation history.**
2.  **Decide the next step.**
    -   **Case 1: The query is a simple greeting or off-topic.** Respond conversationally.
    -   **Case 2: The query is about a legal case but is missing critical details.** Before using the `analyze_case` tool, you MUST check if you have the following information:
        - For dismissal cases: The reason for dismissal, contract type (limited/unlimited), and if any warnings were issued.
        - For leave cases: The **exact duration of service** (e.g., "2 years", "6 years") to determine the correct leave entitlement (3 or 4 weeks).
        - For end-of-service gratuity cases: The contract type and total duration of service.
        If any of this critical information is missing, your goal is to ask a clarifying question.
    -   **Case 3: The query is a complete and detailed legal case that passes the checklist above.** Your goal is to use the `analyze_case` tool.

3.  **Formulate your response based on your decision.**

Use the following format:

Question: the input question you must answer
Thought: (Your reasoning process in English). I need to decide if I have enough information to use the tool, or if I need to ask a question based on my checklist.
    - If you decide to ask a question:
        Thought: The user's query is incomplete. I need to ask for more details about [specific missing information from the checklist]. I will formulate my question in Arabic and provide it as the Final Answer.
        Final Answer: [Your clarifying question in Arabic]
    - If you decide to use the tool:
        Thought: The user has provided a complete case that satisfies my checklist. I will use the `analyze_case` tool to get a full legal analysis.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: [The full, detailed query from the user in Arabic]
        Observation: [The result from the `analyze_case` tool]
        Thought: I have the analysis from the tool. I will now format it and provide it as the final answer.
        Final Answer: [The complete legal analysis from the Observation, presented clearly in Arabic]

**Important Rules:**
- Your `Final Answer` MUST always be in Arabic.
- If you ask a question, do NOT use a tool. Just provide the `Final Answer`.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

agent_prompt = PromptTemplate.from_template(agent_prompt_template)

# 3. إنشاء الوكيل
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- مسارات (Routes) تطبيق الويب ---

@app.route('/')
def index():
    """عرض الصفحة الرئيسية ومسح سجل المحادثة."""
    session['chat_history'] = []
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """معالجة رسالة المستخدم كجزء من محادثة."""
    try:
        user_query = request.json.get('message')
        if not user_query:
            return jsonify({"error": "No message provided"}), 400

        chat_history_tuples = session.get('chat_history', [])
        chat_history = [HumanMessage(content=t[0]) if t[1] == 'user' else AIMessage(content=t[0]) for t in chat_history_tuples]

        response = agent_executor.invoke({
            "input": user_query,
            "chat_history": chat_history
        })

        ai_response = response['output']

        chat_history_tuples.append((user_query, 'user'))
        chat_history_tuples.append((ai_response, 'ai'))
        session['chat_history'] = chat_history_tuples

        return jsonify({"response": ai_response})

    except Exception as e:
        print("--- ❌ An Error Occurred ---")
        print(traceback.format_exc())
        print("---------------------------")
        return jsonify({"error": f"حدث خطأ في الخادم: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)