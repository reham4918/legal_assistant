# -*- coding: utf-8 -*-
"""
مساعد قانوني ذكي بواجهة ويب باستخدام Flask
الاستخدام:
  1) تأكد من وجود ملف .env يحتوي على مفتاح Groq API.
  2) تأكد من وجود قاعدة بيانات ChromaDB في مجلد 'chroma_db'.
  3) قم بتشغيل هذا التطبيق: flask --app app run
  4) افتح المتصفح على العنوان: http://127.0.0.1:5000
"""
from flask import Flask, render_template, request
from assistant_core import LegalAssistant

# --- تهيئة تطبيق Flask ---
app = Flask(__name__)

# --- تحميل وتهيئة المساعد القانوني (يتم مرة واحدة عند بدء التشغيل) ---
# الكلاس LegalAssistant يقوم بكل عمليات التحميل والتهيئة المعقدة
legal_assistant = LegalAssistant()


# --- مسارات (Routes) تطبيق الويب ---

@app.route('/')
def index():
    """عرض الصفحة الرئيسية التي تحتوي على نموذج الإدخال."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """معالجة طلب التحليل من المستخدم."""
    user_query = request.form.get('query')
    if not user_query:
        return render_template('index.html', error="الرجاء إدخال نص القضية.")

    # --- المنطق الأساسي للتحليل أصبح الآن استدعاءً لدالة واحدة ---
    # هذا يجعل الكود أكثر نظافة وسهولة في الصيانة
    analysis_result = legal_assistant.analyze_case(user_query)

    # استبدال علامات السطر الجديد بـ <br> لعرضها بشكل صحيح في HTML
    formatted_analysis = analysis_result.replace('\n', '<br>')
    return render_template('results.html', query=user_query, analysis=formatted_analysis)


if __name__ == '__main__':
    # لتشغيل التطبيق، استخدم الأمر: flask --app app run
    # أو قم بتشغيله كأي ملف بايثون عادي
    app.run(debug=True)