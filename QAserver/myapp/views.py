# from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework.authtoken.views import APIView, AuthTokenSerializer
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.decorators import api_view
from chatbot import Chat
from django.http import JsonResponse
from django.views import View
import json


# Create your views here.

# 用户注册


class Register(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        password = request.data.get('password')
        if User.objects.filter(username=username).exists():
            resp = {
                'status': False,
                'data': '用户名已被注册'
            }
        else:
            user = User.objects.create_user(
                username=username, password=password)
            token, created = Token.objects.get_or_create(user=user)
            resp = {
                'status': True,
                'token': token.key,
                'user_id': user.pk,
                'user_name': user.username,
            }
        return Response(resp)

# 用户登录


class Login(APIView):

    def post(self, request, *args, **kwargs):
        serializer = AuthTokenSerializer(
            data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'status': True,
            'token': token.key,
            'user_id': user.pk,
            'user_name': user.username,
        })


@api_view(['GET'])
def answer_api(request):
    res_msg = {'data': "112"}
    res_msg['data'] = 111
    return Response(res_msg)


@api_view(['POST'])
def question_api(request):
    chatview = Chat()
    res_msg = {'test': "hello"}
    json_str = request.POST("")
    print(json_str)
    ans = Chat.chat_main(json_str)
    res_msg['test'] = ans
    return Response(res_msg)


class QAView(APIView):

    # def get(self, request):
    #     response_msg = {'status': 100, 'data': 'hello'}
    #     response_msg['status'] = 200
    #     response_msg['test'] = 'hello'
    #     return Response(response_msg)

    def post(self, request):
        # print(request.META, sep='\n')
        # print(request.POST, sep='\n')
        js = request
        json_str= js.dumps(request)
        # print(json_str, sep='\n')
        # json_str = json_str.decode("utf-8")
        # print(json_str, sep='\n')
        # sent = json.loads(json_str)
        print(sent)
        # chat = Chat()
        # ans = chat.chat_main(sent)
        ans = '测试数据'
        res_msg = {'test': "hello"}
        res_msg["test"] = ans
        return Response(res_msg)
