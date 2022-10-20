from django.shortcuts import render

ice_cream_catalog = [
    {
        'id': 0,
        'title': 'Пивное мороженое',
        'description': 'Мечта любого пивомана: мороженое '
                       'со вкусом медового пива и темного эля.',
    },
    {
        'id': 1,
        'title': 'Мороженое с кузнечиками',
        'description': 'В колумбийском стиле: мороженое '
                       'с добавлением настоящих карамелизованных кузнечиков.',
    },
    {
        'id': 2,
        'title': 'Мороженое со вкусом сыра чеддер',
        'description': 'Вкус настоящего сыра в вафельном стаканчике.',
    },
]


def ice_cream_detail(request, pk):
    template = 'ice_cream/detail.html'
    context = {'ice_cream': ice_cream_catalog[pk]}
    return render(request, template, context)


def ice_cream_list(request):
    template = 'ice_cream/list.html'
    context = {'ice_cream_list': ice_cream_catalog}
    return render(request, template, context)
