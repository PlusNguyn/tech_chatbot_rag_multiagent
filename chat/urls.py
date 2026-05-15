from django.urls import path

from chat.views import chat_message


urlpatterns = [
    path("message/", chat_message, name="chat-message"),
]

