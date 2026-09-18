# C:/Users/LENOVO/PycharmProjects/legal_assistant/app.py

# -*- coding: utf-8 -*-
"""
مساعد قانوني ذكي بواجهة ويب باستخدام Flask
الاستخدام:
  1) تأكد من وجود ملف .env يحتوي على مفتاح OpenRouter API.
  2) تأكد من وجود قاعدة بيانات ChromaDB في مجلد 'chroma_db'.
  3) قم بتشغيل هذا التطبيق: python app.py
"""
from flask import Flask, render_template, request
import traceback
import markdown2  # ✨ استيراد جديد لمعالجة الماركدوان

# --- تهيئة تطبيق Flask ---
app = Flask(__name__)

# --- محاولة تهيئة المساعد القانوني ---
# سنحاول استيراد وتهيئة المساعد، وإذا فشل، سنلتقط الخطأ
try:
    from assistant_core import LegalAssistant

    print("--- [APP START] جاري تهيئة المساعد القانوني. قد يستغرق هذا بعض الوقت... ---")
    legal_assistant = LegalAssistant()
    print("--- [APP START] ✅ تم تهيئة المساعد بنجاح. التطبيق جاهز. ---")
except Exception as e:
    print(f"--- ❌ [خطأ فادح] فشل تهيئة المساعد القانوني أثناء بدء التشغيل ---")
    print(traceback.format_exc())
    print("----------------------------------------------------------------")
    legal_assistant = None


# --- مسارات (Routes) تطبيق الويب ---

@app.route('/')
def index():
    """عرض الصفحة الرئيسية التي تحتوي على نموذج الإدخال."""
    if legal_assistant is None:
        error_message = "فشل تحميل المكونات الأساسية للمساعد القانوني. يرجى مراجعة سجلات الخادم (الطرفية/console) للمزيد من التفاصيل."
        # تأكد من وجود ملف error.html في مجلد templates
        return render_template('error.html', error_message=error_message), 500
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """معالجة طلب التحليل من المستخدم."""
    if legal_assistant is None:
        error_message = "المساعد القانوني غير متوفر بسبب خطأ في التهيئة. يرجى مراجعة سجلات الخادم."
        return render_template('error.html', error_message=error_message), 500

    user_query = request.form.get('query')
    if not user_query:
        return render_template('index.html', error="الرجاء إدخال نص القضية.")

    analysis_result = legal_assistant.analyze_case(user_query)

    # ✨ تعديل: تحويل النص من Markdown إلى HTML
    # هذا سيقوم بتحويل **النص** إلى <strong>النص</strong> والأسطر الجديدة إلى <br>
    # وأيضًا القوائم النقطية (*) إلى <ul><li>...</li></ul>
    html_analysis = markdown2.markdown(analysis_result, extras=["fenced-code-blocks", "tables"])

    return render_template('results.html', query=user_query, analysis=html_analysis)


if __name__ == '__main__':
    app.run(debug=True)