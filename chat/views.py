import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from chat.services import ask_chatbot


def home_view(request):
    return JsonResponse(
        {
            "service": "Tech Chatbot RAG Multi-Agent",
            "status": "ok",
            "routes": {
                "chat_page": "/chat/",
                "chat_api": "/chat/message/",
                "admin": "/admin/",
            },
        }
    )


@csrf_exempt
@require_POST
def chat_message(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    query = str(payload.get("query", "")).strip()
    if not query:
        return JsonResponse({"error": "Field 'query' is required."}, status=400)

    try:
        return JsonResponse(ask_chatbot(query))
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    

def chat_view(request):
    return render(request, "chat/chat.html")
