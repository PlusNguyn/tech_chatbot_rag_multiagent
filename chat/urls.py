from django.urls import path

from chat.views import chat_message, chat_view


urlpatterns = [
    path("", chat_view, name="home"),
    path("chat/", chat_view, name="chat-view"),
    path("message/", chat_message, name="chat-message"),
]
