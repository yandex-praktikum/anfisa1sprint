from django.shortcuts import render


def about(request):
    return render(request, 'pages/about.html')


def index(request):
    return render(request, 'pages/index.html')


def contact(request):
    return render(request, 'pages/contact.html')


def rules(request):
    return render(request, 'pages/rules.html')
