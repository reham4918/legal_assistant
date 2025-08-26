# -*- coding: utf-8 -*-
"""
الاستخدام:
  1) تأكد من وجود ملف .env يحتوي على مفتاح Groq API الخاص بك: GROQ_API_KEY="gsk_..."
  2) قم بتثبيت المكتبات اللازمة: pip install sentence-transformers torch chromadb python-dotenv groq
  3) قم بتشغيل هذا الكود: python step4_ask.py
"""
from assistant_core import LegalAssistant


def main():
    """
    الدالة الرئيسية: تستخدم كلاس المساعد القانوني لتشغيل واجهة سطر الأوامر.
    """
    print(f"\n--- ⚖️ مساعد قانون العمل القطري الذكي ⚖️ ---")

    # --- تهيئة المساعد القانوني ---
    # الكلاس LegalAssistant يقوم بكل عمليات التحميل والتهيئة المعقدة
    # يتم استدعاؤه مرة واحدة هنا
    try:
        assistant = LegalAssistant()
    except ValueError as e:
        print(f"❌ خطأ في التهيئة: {e}")
        return

    print("\nادخل وصف الحالة، أو اكتب 'خروج' لإنهاء البرنامج.")
    print("----------------------------------------------------------\n")

    while True:
        user_input = input("📄 ادخل وصف الحالة للتحليل: ")
        if user_input.lower() in ['خروج', 'exit', 'quit']:
            print("👋 مع السلامة!")
            break
        if not user_input.strip():
            continue

        # --- المنطق الأساسي للتحليل أصبح الآن استدعاءً لدالة واحدة ---
        # هذا يجعل الكود أكثر نظافة وسهولة في الصيانة
        analysis = assistant.analyze_case(user_input)

        print("\n" + "=" * 25 + " تحليل الحالة " + "=" * 25)
        print(analysis)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()