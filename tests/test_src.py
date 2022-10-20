import difflib
import inspect
import os
import re
import sys
from collections import defaultdict
from functools import partial
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest

from itertools import zip_longest, chain

from yap_testing.common import (
    YapTestingException, format_nested,
    partial_format_simple, multiple_replace)
from yap_testing.compare_diff import (
    compare_file_to_author,
    read_files_asserting_content, CompareLinesDiffWithAuthor)
from yap_testing.decline_dict.decline_dict import DECLINE_DICT_FLAT
from yap_testing.html.utils import comments_re, removed, added, DiffGetter
from yap_testing.output import CapturingStdout
from yap_testing.run_pytest import run_pytest_runner, PytestRunner
from yap_testing.webpack.pytest_webpack import PytestFileWebpacker

import_dependencies = (
    f'{PytestRunner}',  # include in every test file
    f'{CapturingStdout}',  # include in every test file
    f'{sys}',  # include in every test file
    f'{StringIO}',  # include in every test file
    f'{DECLINE_DICT_FLAT}',
    f'{format_nested}',
    f'{partial_format_simple}',
    f'{ModuleType}',
    f'{inspect}',
    f'{multiple_replace}',
    f'{read_files_asserting_content}',
    f'{re}',
    f'{zip_longest}'
    f'{CompareLinesDiffWithAuthor}',
    f'{difflib}',
    f'{removed}',
    f'{added}',
    f'{chain}',
    f'{partial}',
    f'{DiffGetter}',
    f'{defaultdict}',
)


@pytest.fixture(scope='session', autouse=True)
def is_in_production():
    return True


def get_dir_from_template(path_template: str, root: str):
    assert path_template.startswith('{}/')
    if root:
        root = str(root)
        if root == '/':
            root = ''
        elif root.endswith('/'):
            root = root[:-1]
    return Path(path_template.format(root)).resolve()


@pytest.fixture(scope='module')
def header_path_template():
    return '{}/templates/includes/header.html'


@pytest.fixture(scope='module')
def base_path_template():
    return '{}/templates/base.html'


@pytest.fixture(scope='module')
def list_path_template():
    return '{}/templates/ice_cream/list.html'


@pytest.fixture(scope='module')
def detail_path_template():
    return '{}/templates/ice_cream/detail.html'


@pytest.fixture(scope='module')
def settings_path_template():
    return '{}/anfisa_for_friends/settings.py'


@pytest.fixture(scope='module')
def lesson_dir(is_in_production):
    return str(Path(__file__).parent.parent)


@pytest.fixture(scope='module')
def student_dir(lesson_dir, is_in_production):
    if Path(__file__).name == 'test_src.py':
        # run in `author` dir, test same dir
        return Path(__file__).parent.as_posix()
    else:
        # run in `tests` dir, test project (aka "root") dir
        return Path(__file__).parent.parent.as_posix()


@pytest.fixture
def precode_dir(lesson_dir):
    return (Path(lesson_dir) / 'precode/').as_posix()


@pytest.fixture
def author_dir(lesson_dir):
    return (Path(lesson_dir) / 'author/').as_posix()


def header_html_path(dirname: str, header_path_template) -> Path:
    path = get_dir_from_template(header_path_template, dirname)
    return path


def base_html_path(dirname: str, base_path_template) -> Path:
    path = get_dir_from_template(base_path_template, dirname)
    return path


def list_html_path(dirname: str, list_path_template) -> Path:
    path = get_dir_from_template(list_path_template, dirname)
    return path


def detail_html_path(dirname: str, detail_path_template) -> Path:
    path = get_dir_from_template(detail_path_template, dirname)
    return path


def settings_path(dirname: str, settings_path_template) -> Path:
    path = get_dir_from_template(settings_path_template, dirname)
    return path


def test_header_html(
        student_dir, precode_dir, author_dir, header_path_template,
        is_in_production):
    student_path = header_html_path(student_dir, header_path_template)
    precode_path = header_html_path(precode_dir, header_path_template)
    author_path = header_html_path(author_dir, header_path_template)

    precode_hardcode = """{% load static %}
<header>
  <img src="{% static 'img/logo.png' %}" height="50" alt="">
  <nav>
    <ul>
      <li>
        <a href="{% url 'homepage:index' %}">Главная</a>  
      </li>
      <li>
        <a href="{% url 'ice_cream:ice_cream_list' %}">Список мороженого</a>       
      </li>
      <li>
        <a href="{% url 'about:description' %}">О проекте</a>
      </li>
    </ul>
  </nav>
</header>"""
    author_hardcode = """{% load static %}
<header>
  <nav class="navbar navbar-expand-lg navbar-light bg-light fixed-top shadow-sm">
    <div class="container">
      <div
        class="navbar-toggler"
        type="button"
        data-toggle="collapse"
        data-target="#navbarNav"
        aria-controls="navbarNav"
        aria-expanded="false"
        aria-label="Toggle navigation"
        >
        <span class="navbar-toggler-icon"></span>
      </div>
      <div class="collapse navbar-collapse" id="navbarNav">
        {% with request.resolver_match.view_name as view_name %}
        <ul class="nav nav-pills">
          <li class="nav-item">
            <a
              class="nav-link {% if view_name == 'homepage:index' %} active {% endif %}"
              href="{% url 'homepage:index' %}"
              >
              Главная
            </a> 
          </li>
          <li class="nav-item">
            <a
              class="nav-link {% if view_name == 'ice_cream:ice_cream_list' %} active {% endif %}"
              href="{% url 'ice_cream:ice_cream_list' %}"
              >
              Список мороженого
            </a> 
          </li>
          <li class="nav-item">
            <a
              class="nav-link {% if view_name == 'about:description' %} active {% endif %}"
              href="{% url 'about:description' %}"
            >
              О проекте
            </a> 
          </li>
        </ul>
        {% endwith %}
      </div>
      <a class="navbar-brand" href="{% url 'homepage:index' %}">
        <img src="{% static 'img/logo.png' %}" height="50" class="d-inline-block align-top" alt="">
      </a>
    </div>
  </nav>
</header>"""
    try:
        compare_file_to_author(
            student_path, author_path, precode_path, precode_hardcode,
            author_hardcode, remove_pattern=comments_re(), strip=True,
            compare_n_items_changed=False,
            assert_file_content_unchanged=not is_in_production
        )
    except YapTestingException as e:
        assert False, f'Ошибка в файле `{student_path.relative_to(student_path.parent.parent.parent)}`:\n{str(e)}'


def test_base_html(
        student_dir, precode_dir, author_dir, base_path_template,
        is_in_production):
    student_path = base_html_path(student_dir, base_path_template)
    precode_path = base_html_path(precode_dir, base_path_template)
    author_path = base_html_path(author_dir, base_path_template)

    expect_precode = (
        """{% load static %}
<!DOCTYPE html>
<html lang="ru">
  <head>
    <link rel="icon" href="{% static 'img/fav/fav.ico' %}" type="image">
    <link rel="apple-touch-icon" sizes="180x180" href="{% static 'img/fav/apple-touch-icon.png' %}">
    <link rel="icon" type="image/png" sizes="32x32" href="{% static 'img/fav/favicon-32x32.png' %}">
    <link rel="icon" type="image/png" sizes="16x16" href="{% static 'img/fav/favicon-16x16.png' %}">
    <title>
      {% block title %}{% endblock %}
    </title>
    <link rel="stylesheet" href="{% static 'css/bootstrap.min.css' %}">
  </head>
  <body>
    {% include "includes/header.html" %}
    <main>
      {% block content %}{% endblock %}
    </main>
    {% include "includes/footer.html" %}
  </body>
</html>""")
    expect_author = """{% load static %}
<!DOCTYPE html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" href="{% static 'img/fav/fav.ico' %}" type="image">
    <link rel="apple-touch-icon" sizes="180x180" href="{% static 'img/fav/apple-touch-icon.png' %}">
    <link rel="icon" type="image/png" sizes="32x32" href="{% static 'img/fav/favicon-32x32.png' %}">
    <link rel="icon" type="image/png" sizes="16x16" href="{% static 'img/fav/favicon-16x16.png' %}">
    <title>
      {% block title %}{% endblock %}
    </title>
    <link rel="stylesheet" href="{% static 'css/bootstrap.min.css' %}">
  </head>
  <body>
    {% include "includes/header.html" %}
    <main class="pt-5 my-5">
      <div class="container">
      {% block content %}{% endblock %}
      </div>
    </main>
    {% include "includes/footer.html" %}
  </body>
  <script src="{% static 'js/jquery-3.2.1.slim.min.js' %}"></script>
  <script src="{% static 'js/bootstrap.min.js' %}"></script>
</html>"""
    try:
        compare_file_to_author(
            student_path, author_path, precode_path, expect_precode,
            expect_author, remove_pattern=comments_re(), strip=True,
            compare_n_items_changed=False,
            assert_file_content_unchanged=not is_in_production
        )
    except YapTestingException as e:
        assert False, f'Ошибка в файле `{student_path.relative_to(student_path.parent.parent.parent)}`:\n{str(e)}'


def test_list_html(
        student_dir, precode_dir, author_dir, list_path_template,
        is_in_production):
    student_path = list_html_path(student_dir, list_path_template)
    precode_path = list_html_path(precode_dir, list_path_template)
    author_path = list_html_path(author_dir, list_path_template)

    expect_precode = (
        """{% extends "base.html" %}
{% block title %}
  Анфиса для друзей. Список мороженого
{% endblock %}
{% block content %}
  <h1>Список мороженого</h1>
  <ul>
    {% for ice_cream in ice_cream_list %}
      <li>
        <h3>{{ ice_cream.title }}</h3>
        <p>{{ ice_cream.description|truncatewords:10 }}</p>
        <a href="{% url 'ice_cream:ice_cream_detail' ice_cream.id %}">подробнее</a>
      </li>
      {% if not forloop.last %}
        <hr>
      {% endif %}
    {% endfor %}
  </ul>
{% endblock %}""")
    expect_author = """{% extends "base.html" %}
{% load static %}
{% block title %}
  Анфиса для друзей. Список мороженого
{% endblock %}
{% block content %}
<h1 class="border-bottom pb-2 mb-0">Каталог мороженого</h1>
  <div class="row">
    {% for ice_cream in ice_cream_list %}
      <div class="col-6 col-md-4 my-1">
        <div class="card">
          <img
            class="img-fluid card-img-top"
            height="400" width="300"
            src="{% static 'img/image-holder.png' %}"
          >
          <div class="card-body">
            <h5 class="card-title">{{ ice_cream.title }}</h5>
            <p class="card-text">{{ ice_cream.description|truncatewords:10 }}</p>
            <a class="mt-3 regular-link" href="{% url 'ice_cream:ice_cream_detail' ice_cream.id %}">
              Подробнее -->
            </a>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>
{% endblock %}"""
    try:
        compare_file_to_author(
            student_path, author_path, precode_path, expect_precode,
            expect_author, remove_pattern=comments_re(), strip=True,
            compare_n_items_changed=False,
            assert_file_content_unchanged=not is_in_production
        )
    except YapTestingException as e:
        assert False, f'Ошибка в файле `{student_path.relative_to(student_path.parent.parent.parent)}`:\n{str(e)}'


def test_detail_html(
        student_dir, precode_dir, author_dir, detail_path_template,
        is_in_production):
    student_path = detail_html_path(student_dir, detail_path_template)
    precode_path = detail_html_path(precode_dir, detail_path_template)
    author_path = detail_html_path(author_dir, detail_path_template)

    expect_precode = (
        """{% extends "base.html" %}
{% block title %}
  Анфиса для друзей. Подробное описание мороженого
{% endblock %}
{% block content %}
  <h1>Подробное описание мороженого</h1>
  <h3>{{ ice_cream.title }}</h3>
  <p>{{ ice_cream.description }}</p>
{% endblock %}
 """)
    expect_author = """{% extends "base.html" %}
{% load static %}
{% block title %}
  Анфиса для друзей. Мороженое {{ ice_cream.title|lower }}
{% endblock %}
{% block content %}
  <h1 class="border-bottom pb-2 mb-0">{{ ice_cream.title }}</h1>  
  <div class="row mt-3">
    <div class="col-12 col-md-6">
      <h2>Описание</h2>
      <p>{{ ice_cream.description }}</p>      
    </div> 
    <div class="col-12 col-md-6 mb-3">
      <img 
        class="img-fluid" 
        height="400" width="300"
        src="{% static 'img/image-holder.png' %}"
        >      
    </div>
  </div>     
{% endblock %}
 """
    try:
        compare_file_to_author(
            student_path, author_path, precode_path, expect_precode,
            expect_author, remove_pattern=comments_re(), strip=True,
            compare_n_items_changed=False,
            assert_file_content_unchanged=not is_in_production
        )
    except YapTestingException as e:
        assert False, f'Ошибка в файле `{student_path.relative_to(student_path.parent.parent.parent)}`:\n{str(e)}'


ignore_dependencies = {
    f'{str(PytestFileWebpacker)}',
    'import_dependencies'
}

if __name__ == '__main__':
    if 'SKIP_WEBPACK' in os.environ:
        run_pytest_runner(__file__)
    else:  # webpack for production
        webpacked_fpath = PytestFileWebpacker(
            __file__,
            remove_target_file=True,
            include_runner_code=True
        ).webpack_py_file()
        run_pytest_runner(webpacked_fpath)
