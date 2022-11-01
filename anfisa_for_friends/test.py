# -*- coding: utf-8 -*-

import difflib
import os
import re
import sys
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pytest


def read_text_file_asserting_content(
        file_path: Union[str, Path], expect_content: Optional[str],
        remove_re_when_comparing: Union[str, object],
        expect_content_not_set_warning=(
                '\n\nProvide expected content for file '
                '\n`{file_path}`\n'
                'Here is the code to set it '
                '(DO NOT INDENT THE TEXT BETWEEN TRIPLE QUOTES):\n'
                'expect_content="""{current_content}"""'
        ),
        unexpected_content_msg=(
                'File \n`{file_path}`\n '
                'content is not as expected (has been changed?): '
                '`\n'
                '{diff}`\n'
                '\nUpdate the expected value to the following:\n\n'
                '"""{current_content}"""'
        ),
        strip_lines=False
):
    """If existing `file_path` is provided, checks that its content is as
    expected and returns it, otherwise raises ValueError.
    """
    if file_path:
        if not os.path.isfile(file_path):
            raise YapTestingException(
                f'Не найден файл `{file_path}`.')
    else:
        return expect_content

    def clear_content(content_):
        cleared_ = content_
        if remove_re_when_comparing:
            cleared_ = re.sub(remove_re_when_comparing, '', content_)
        if strip_lines:
            cleared_ = '\n'.join(map(str.strip, cleared_.split('\n')))
        return cleared_

    with open(file_path, 'r', encoding='utf8') as fh:
        current_content = fh.read()
        current_content_cleared = clear_content(current_content)
        if expect_content:
            expect_content_cleared = clear_content(expect_content)
            if current_content_cleared != expect_content_cleared:
                diff = '\n'.join(difflib.ndiff(
                    expect_content_cleared.split('\n'),
                    current_content_cleared.split('\n')))
                raise ValueError(
                    partial_format_simple(
                        unexpected_content_msg, **locals())
                )
        elif expect_content_not_set_warning:
            print(partial_format_simple(
                expect_content_not_set_warning, **locals()))

    return current_content


@dataclass
class DiffsToMake:
    text: str = ''
    line_pos: int = 1


class CompareDiffWithAuthor(ABC):
    """Check that diff between author and precode is the same as between
    student and precode for:
    """

    def __init__(
            self, student_path: Union[Path, str], **kwargs
    ):
        self.student_path = student_path

        self._student_items: Optional[list] = None
        self._author_items: Optional[list] = None
        self._precode_items: Optional[list] = None
        self._diff_precode_student: Optional[list] = None
        self._diff_precode_author: Optional[list] = None
        self._diff_author_student: Optional[list] = None

        # the following are set when a difference in solutions
        # is narrowed down to a specific item
        self.author_item = None
        self.student_item = None
        self.diffs_to_make = None

    @property
    @abstractmethod
    def checked_item_name(self):
        """The name of the kind of items being compared for use in
        except messages"""
        raise NotImplementedError

    def decline(self, text, apply_dicts: Union[dict, Iterable[dict]]):
        if isinstance(apply_dicts, dict):
            apply_dicts = [apply_dicts]
        else:
            apply_dicts = list(apply_dicts)
        apply_dicts.insert(0, {'checked_item_name': self.checked_item_name})
        apply_dicts.append(DECLINE_DICT_FLAT)
        first_go = format_nested(text, apply_dicts)
        # transform expressions like
        # {значение+переменный_femn_|datv_plur},
        # into
        # {значение|datv_plur} {переменный|femn_gent_plur}
        # for further formatting
        pairs_separated = re.sub(r'{([А-я]+)\+([А-я]+)_*(\w*_*)\|(\w+)_(\w+)}',
                                 r'{\1|\4_\5} {\2|\3gent_\5}', first_go)
        second_go = format_nested(pairs_separated, apply_dicts)
        return second_go[:1].capitalize() + second_go[1:]

    @abstractmethod
    def make_items(self, *args, **kwargs):
        raise NotImplementedError

    def _calc_diffs(self):
        self._diff_precode_author = list(
            difflib.ndiff(self.precode_items, self.author_items))
        self._diff_precode_student = list(
            difflib.ndiff(self.precode_items, self.student_items))
        self._diff_author_student = list(
            difflib.ndiff(self.author_items, self.student_items))

    @property
    def diff_precode_author(self):
        if self._diff_precode_author is None:
            self._calc_diffs()
        return self._diff_precode_author

    @property
    def diff_precode_student(self):
        if self._diff_precode_student is None:
            self._calc_diffs()
        return self._diff_precode_student

    @property
    def diff_author_student(self):
        if self._diff_author_student is None:
            self._calc_diffs()
        return self._diff_author_student

    @property
    def author_items(self):
        if self._author_items is None:
            assert False, 'Call make_items before accessing the property'
        return self._author_items

    @author_items.setter
    def author_items(self, v):
        self._author_items = v

    @property
    def student_items(self):
        if self._student_items is None:
            assert False, 'Call make_items before accessing the property'
        return self._student_items

    @student_items.setter
    def student_items(self, v):
        self._student_items = v

    @property
    def precode_items(self):
        if self._precode_items is None:
            assert False, 'Call make_items before accessing the property'
        return self._precode_items

    @precode_items.setter
    def precode_items(self, v):
        self._precode_items = v

    @property
    def match_n_items_msg(self):
        """
        Template of message for student.
        Redefined in descendant classes to make it human-readable.
        """
        return (
            f'Убедитесь, что количество '
            '{{checked_item_name}|gent_plur}, '
            '{{operated}|gent_plur}/{{operated_or}|gent_plur} '
            'в прекоде файла'
            '\n`{student_file}`\n'
            ', равно {operated_n_author} '
            '(строки комментариев не учитываются).\n'
            'В настоящий момент число '
            '{{operated}|gent_plur}/{{operated_or}|gent_plur} '
            '{{checked_item_name}|gent_plur} в прекоде равно '
            '{operated_n_student}:\n'
            '`\n{diff}\n`'
        )

    @property
    def how_fix_msg(self):
        """
        Template of message for student.
        Redefined in descendant classes to make it human-readable.
        """
        if self.student_item:
            return (
                'Попробуйте вставить {{checked_item_name}|accs_sing} '
                'перед, после или вместо {{checked_item_name}|gent_sing} '
                '\n\n`{student_item}`\n\n'
                'В последнем случае в {{checked_item_name}|accs_sing} '
                'необходимо внести следующие изменения '
                'после строки прекода №{diffs_to_make_line_pos} '
                '(строки комментариев не учитываются): '
                '\n'
                '`\n{diffs_to_make_text}\n`'
                '(где `-` - удалить, `+` - добавить, `?` - позиция различий).'
            )
        else:
            return (
                'Изменения необходимо внести '
                'после строки прекода №{diffs_to_make_line_pos} '
                '(строки комментариев не учитываются).'
            )

    @property
    def remove_or_edit_item_msg(self):
        """
        Template of message for student.
        Redefined in descendant classes to make it human-readable.
        """
        return (
            '{{checked_item_name}|gent_sing} '
            '\n`{remove_item}`\n'
            'не должно быть в решении. '
            'Убедитесь, что в {{checked_item_name}|loct_sing} нет ошибок '
            'и что {{checked_item_name}|nomn_sing} не дублируется.\n'
            '{how_fix}'
        )

    @property
    def add_or_edit_item_msg(self):
        """
        Template of message for student.
        Redefined in descendant classes to make it human-readable.
        """
        return (
            'Убедитесь, что {{checked_item_name}|nomn_sing} '
            '\n`{add_item}` \n'
            'присутствует в решении, и что в '
            '{{checked_item_name}|loct_sing} нет ошибок.\n'
            '{how_fix}'
        )

    def compare_with_author(self):
        """
        Sequentially calculates added/removed items (diffs)
        in author and student and checks differences in the two diffs.
        If differences are found, YapTestingException is raised with a message
        showing the difference.
        """
        for (
                changed,  # callback looking up differences
                except_if_diff  # callback raising exception if diff found
        ) in (
                (
                        # removed items checked 1st as they come first in
                        # `difflib.ndiff` output
                        partial(removed, only_1st_diffs=False),
                        self.except_if_removed_diff
                ),
                (
                        partial(added, only_1st_diffs=False),
                        self.except_if_added_diff
                ),
        ):
            changed_author, changed_author_ixs = changed(
                self.diff_precode_author, diff_from=self.precode_items)
            changed_student, changed_student_ixs = changed(
                self.diff_precode_student, diff_from=self.precode_items)
            self.compare_diff_with_author(
                changed_author, changed_author_ixs, changed_student,
                changed_student_ixs,
                except_if_diff=except_if_diff
            )

    def compare_diff_with_author(
            self, changed_author: list, changed_author_ixs: list,
            changed_student: list, changed_student_ixs: list,
            except_if_diff: Callable
    ) -> None:
        """Takes changed (only added/removed) items in author and student,
        aligns them on position they were added to/removed from,
        compares student items to author's and raises for the first
        found difference.
        """

        aligned_keys = dict.fromkeys(sorted(set(
            chain(changed_author_ixs, changed_student_ixs))))
        aligned_author, aligned_student = (
            aligned_keys.copy(), aligned_keys.copy())
        aligned_author.update(dict(zip(changed_author_ixs, changed_author)))
        aligned_student.update(dict(zip(changed_student_ixs, changed_student)))

        zip_author_student = zip(
            aligned_author.values(), aligned_student.values())
        zip_author_student_ixs = zip(
            aligned_author.keys(), aligned_student.keys())

        author_item, student_item, diffs_to_make = self._get_first_diff(
            zip_author_student, zip_author_student_ixs)
        except_if_diff(author_item, student_item, diffs_to_make)

    def except_if_added_diff(
            self, author_item, student_item, diffs_to_make: DiffsToMake):
        """Compile message and raise exception with if items added to precode
        differ in student and author."""
        self.author_item = author_item
        self.student_item = student_item
        self.diffs_to_make = diffs_to_make
        if author_item:
            except_msg = self.add_or_edit_item_msg
        elif student_item:
            except_msg = self.remove_or_edit_item_msg
        else:
            except_msg = ''

        how_fix = self.how_fix_msg if diffs_to_make else ''

        if except_msg:
            decline_args = {
                'add_item': author_item,
                'remove_item': student_item,
                'how_fix': how_fix,
                'diffs_to_make_text': diffs_to_make.text,
                'diffs_to_make_line_pos': str(diffs_to_make.line_pos - 1),
                'checked_item_name': self.checked_item_name,
                'student_item': student_item,
                'author_item': author_item,
            }
            except_msg = self.decline(except_msg, decline_args)
            raise YapTestingException(except_msg)

    def except_if_removed_diff(
            self, author_item, student_item, diffs_to_make: DiffsToMake):
        """Compile message and raise exception with if items removed from
        precode differ in student and author."""
        if author_item:
            if student_item:
                except_msg = self.remove_or_edit_item_msg
            else:
                except_msg = self.remove_or_edit_item_msg
        elif student_item:
            except_msg = self.add_or_edit_item_msg
        else:
            except_msg = ''

        if except_msg:
            decline_args = {
                'add_item': student_item,  # incorrectly removed
                'remove_item': author_item,  # should have been removed
                'how_fix': (
                    'Изменения необходимо внести '
                    'после строки прекода №{diffs_to_make_line_pos} '
                    '(строки комментариев не учитываются).'
                ),
                'diffs_to_make_text': diffs_to_make.text,
                'diffs_to_make_line_pos': diffs_to_make.line_pos - 1,
                'checked_item_name': self.checked_item_name,
            }
            except_msg = self.decline(except_msg, decline_args)
            raise YapTestingException(except_msg)

    def compare_n_items_changed(self):
        """Checks that number of items added to/removed from precode is equal
         for author and student solutions."""
        author_added = added(
            self.diff_precode_author, diff_from=self.precode_items)[0]
        student_added = added(
            self.diff_precode_student, diff_from=self.precode_items)[0]
        if len(student_added) != len(author_added):
            diff = '\n'.join(self.diff_precode_student)
            raise YapTestingException(
                self.decline(self.match_n_items_msg,
                             {
                                 'operated': 'добавленный',
                                 'operated_or': 'изменённый',
                                 '{добавленных/удалённых}': 'добавленных',
                                 'student_file': self.student_path,
                                 'operated_n_author': len(author_added),
                                 'operated_n_student': len(student_added),
                                 'diff': diff,
                             }))

        author_removed = removed(
            self.diff_precode_author, diff_from=self.precode_items)[0]
        student_removed = removed(
            self.diff_precode_student, diff_from=self.precode_items)[0]
        if len(student_removed) != len(author_removed):
            diff = '\n'.join(self.diff_precode_student)
            raise YapTestingException(
                self.decline(self.match_n_items_msg,
                             {
                                 'operated': 'удалённый',
                                 'operated_or': 'изменённый',
                                 '{добавленных/удалённых}': 'удалённых',
                                 'student_file': self.student_path,
                                 'operated_n_author': len(author_removed),
                                 'operated_n_student': len(student_removed),
                                 'diff': diff,
                             }))

    def _check_student_item_eq_author(self, author_item, student_item):
        """Checks that item in student code
        (i.e. line, variable name or variable value) is the same as in
        author's"""
        return author_item == student_item

    def _get_first_diff(self, zip_author_student, zip_author_student_ixs
                        ) -> Tuple[str, str, DiffsToMake]:
        """Finds first differing item aligned on index (position in precode).
        Returns differing items and changes needed to be made in student item.
        """
        diffs_to_make = DiffsToMake()
        for (author_item, student_item), (author_ix, student_ix) in zip(
                zip_author_student, zip_author_student_ixs):
            if not self._check_student_item_eq_author(
                    author_item, student_item):
                if author_ix == student_ix:
                    author_item = author_item or ''
                    student_item = student_item or ''
                    diffs_to_make.text = '\n'.join(
                        filter(lambda x: len(x) > 2,
                               difflib.ndiff(
                                   student_item.split('\n'),
                                   author_item.split('\n'))))
                diffs_to_make.line_pos = min(author_ix, student_ix)
                return author_item, student_item, diffs_to_make
        author_item = student_item = ''
        return author_item, student_item, diffs_to_make


def comments_re():
    return re.compile(r'\s*<!--[\w\W]+?-->')


@pytest.fixture(scope='module', autouse=True)
def update_pythonpath(student_dir):
    os.environ['PYTHONPATH'] = student_dir


@pytest.fixture(scope='module')
def student_dir(lesson_dir, is_in_production):
    result = Path(__file__).parent.as_posix()
    return result


@pytest.fixture(scope='module')
def settings_path_template():
    return '{}/anfisa_for_friends/settings.py'


def settings_path(dirname: str, settings_path_template) -> Path:
    path = get_dir_from_template(settings_path_template, dirname)
    return path


def run_pytest_runner(__file__of_test):
    """
    Run tests from a test file.
    To be called from a test file like so:

    if __name__ == '__main__':
        run_pytest_runner()

    This is how tests must be run in production
    """
    pytest_runner = PytestRunner(__file__of_test)
    pytest_runner.run_capturing_traceback()


def removed(ndiff: list, diff_from: list, only_1st_diffs=False
            ) -> Tuple[List[str], List[int]]:
    diff_getter = DiffGetter(ndiff, diff_from, for_diff_code='- ')
    return diff_getter.get_diff(only_1st_diffs)


def read_files_asserting_content(
        student_path: Union[Path, str],
        author_path: Union[Path, str],
        precode_path: Union[Path, str],
        precode_expected_content: Optional[str],
        author_expected_content: Optional[str],
        remove_pattern: Union[str, object] = comments_re(),
) -> Tuple[str, str, str]:
    """
    Student solution, author solution and precode are read from
    corresponding files given by `...path` variables.
    In production, where author and precode dirs are missing,
    author solution and precode are passed in through `...expected_content`
    variables.
    If file contents in development diverge from `...expected_content`
    variables, an exception is raised.
    """

    student_asserted = read_text_file_asserting_content(
        student_path, expect_content=None,
        remove_re_when_comparing=remove_pattern,
        expect_content_not_set_warning=None, strip_lines=True)

    author_asserted = read_text_file_asserting_content(
        author_path, expect_content=author_expected_content,
        remove_re_when_comparing=remove_pattern, strip_lines=True)

    precode_asserted = read_text_file_asserting_content(
        precode_path, expect_content=precode_expected_content,
        remove_re_when_comparing=remove_pattern, strip_lines=True)

    return student_asserted, author_asserted, precode_asserted


@pytest.fixture
def precode_dir(lesson_dir):
    return (Path(lesson_dir) / 'precode/').as_posix()


def partial_format_simple(s: str, **kwargs) -> str:
    """Formats string `s` without throwing an error if not all keyword
    arguments are provided. In latter case leaves them in the returned
    string unchanged.
    May change formatting in cases where one key is used with different
    formattings, e.g.
    '{var1:1.2f} will be replaced by {var1:1.3f} in this string if `var1` is
    not provided in kwargs'

    >>> partial_format_simple('{var1:1.2f} {var1:1.3f}', var1=2)
    '2.00 2.000'
    >>> partial_format_simple('{var1:1.2f} {var1:1.3f}')
    '{var1:1.3f} {var1:1.3f}'
    >>> partial_format_simple('{var1:1.2f} {var2:1.3f}', var2=2)
    '{var1:1.2f} 2.000'
    >>> partial_format_simple('{var1} {var2:1.3f}', var1='var1')
    'var1 {var2:1.3f}'

    """
    expr_key_pairs = re.findall(r'({([\w\d]+)[^}]*?})', s)
    # e.g. expr_key_pairs = [('{var1:1.2f}', 'var1'), ('{var1:1.3f}', 'var1')]

    # strip all parameters missing from kwargs off their formatting
    s_formats_stripped = multiple_replace(
        s, {k: ('{' + v + '}' if v not in kwargs else k)
            for k, v in expr_key_pairs})

    # make dict to restore formatting where it was changed
    format_params = {v: k for k, v in expr_key_pairs}
    format_params.update(kwargs)

    return s_formats_stripped.format(**format_params)


def multiple_replace(s: str, mapping: dict, as_regex=False) -> str:
    for key in mapping:
        if as_regex or isinstance(key, re.Pattern):
            s = re.sub(key, mapping[key], s)
        else:
            s = s.replace(key, mapping[key])
    return s


@pytest.fixture(scope='module')
def list_path_template():
    return '{}/templates/ice_cream/list.html'


def list_html_path(dirname: str, list_path_template) -> Path:
    path = get_dir_from_template(list_path_template, dirname)
    return path


@pytest.fixture(scope='module')
def lesson_dir(is_building):
    return str(Path(__file__).parent.parent)


@pytest.fixture(scope='session', autouse=True)
def is_in_production(is_building):
    return not is_building


@pytest.fixture(scope='session', autouse=True)
def is_building():
    return 'BUILDING' in os.environ


@pytest.fixture(scope='module')
def header_path_template():
    return '{}/templates/includes/header.html'


def header_html_path(dirname: str, header_path_template) -> Path:
    path = get_dir_from_template(header_path_template, dirname)
    return path


def get_dir_from_template(path_template: str, root: str):
    assert path_template.startswith('{}/')
    if root:
        root = str(root)
        if root == '/':
            root = ''
        elif root.endswith('/'):
            root = root[:-1]
    return Path(path_template.format(root)).resolve()


def format_nested(text: str, apply_dicts: Union[dict, Iterable[dict]]):
    """
    >>> format_nested('{var1|{var2}} {var2}', {'var1': 'v1', 'var2': 'v2'})
    '{var1|v2} v2'

    # order in `apply_dicts` matters:

    >>> format_nested('{var1|{var2}}', [{'var2': 'v2'}, {'var1|v2': 'v3'}])
    'v3'

    >>> format_nested('{var1|{var2}}', [{'{var1|v2}': 'v3'}, {'var2': 'v2'}])
    '{var1|v2}'
    """
    if isinstance(apply_dicts, dict):
        apply_dicts = [apply_dicts]
    for next_dict in apply_dicts:
        for k, v in next_dict.items():
            text = text.replace(f'{{{k}}}', f'{v}')
    return text


@pytest.fixture(scope='module')
def detail_path_template():
    return '{}/templates/ice_cream/detail.html'


def detail_html_path(dirname: str, detail_path_template) -> Path:
    path = get_dir_from_template(detail_path_template, dirname)
    return path


def compare_file_to_author(
        student_path: Union[Path, str], author_path: Union[Path, str],
        precode_path: Union[Path, str],
        precode_expected_content: Optional[str],
        author_expected_content: Optional[str],
        remove_pattern: Union[str, object] = comments_re(),
        strip=False,
        compare_n_items_changed=False,
        assert_file_content_unchanged=True
):
    if assert_file_content_unchanged:
        student_text, author_text, precode_text = read_files_asserting_content(
            student_path, author_path, precode_path, precode_expected_content,
            author_expected_content, remove_pattern)
    else:
        student_text = read_text_file_asserting_content(
            student_path, expect_content=None,
            remove_re_when_comparing=remove_pattern,
            expect_content_not_set_warning=None, strip_lines=True)
        author_text, precode_text = (
            author_expected_content, precode_expected_content)

    if remove_pattern:
        student_items, precode_items, author_items = [
            re.sub(remove_pattern, '', t).split('\n')
            for t in (student_text, precode_text, author_text)
        ]
    else:
        student_items, precode_items, author_items = [
            t.split('\n')
            for t in (student_text, precode_text, author_text)
        ]

    comparer = CompareLinesDiffWithAuthor(student_path)
    comparer.make_items(
        student_items, author_items, precode_items, strip_items=strip)
    if compare_n_items_changed:
        comparer.compare_n_items_changed()
    comparer.compare_with_author()


def comments_re():
    return re.compile(r'\s*<!--[\w\W]+?-->')


@pytest.fixture(scope='module')
def base_path_template():
    return '{}/templates/base.html'


def base_html_path(dirname: str, base_path_template) -> Path:
    path = get_dir_from_template(base_path_template, dirname)
    return path


@pytest.fixture
def author_dir(lesson_dir):
    return (Path(lesson_dir) / 'author/').as_posix()


def added(ndiff: list, diff_from: list, only_1st_diffs=False
          ) -> Tuple[List[str], List[int]]:
    diff_getter = DiffGetter(ndiff, diff_from, for_diff_code='+ ')
    return diff_getter.get_diff(only_1st_diffs)


class YapTestingException(Exception):
    pass


class PytestRunner:
    """
    Run pytest and optionally raise AssertionError with the 1st error
    message from the traceback
    """

    def __init__(self, path: Union[str, Path],
                 args='--exitfirst --disable-warnings',
                 strip_traceback_to_err=True,
                 test_name_contains_expr: Optional[str] = None):
        """
        :param path: path to run pytest on
        :param args: args-string to pass on to pytest
        :param strip_traceback_to_err: search captured traceback for the 1st
            error message and assert False with this message
        :param test_name_contains_expr: e.g. 'test_method',
            'test_method or test_other', 'not test_method and not test_other'
        """
        self.path = path
        self.args = args
        self.strip_traceback_to_err = strip_traceback_to_err
        self.test_name_contains_expr = test_name_contains_expr

    def set_run_args_for_webpack(self):
        self.args = ''
        self.strip_traceback_to_err = False

    @staticmethod
    def clean_msg(msg):
        cleaned = re.sub(r'^E\s+', '', msg, flags=re.MULTILINE)
        cleaned = cleaned.replace('assert False', '')
        cleaned = cleaned.rstrip('\n')
        return cleaned

    def run_capturing_traceback(self, with_args_for_webpack=False) -> None:
        if with_args_for_webpack:
            self.set_run_args_for_webpack()
        with CapturingStdout() as traceback:
            self.args = self.args.split()
            if self.test_name_contains_expr:
                self.args.append(f'-k {self.test_name_contains_expr}')
            self.args.append(str(Path(self.path).as_posix()))
            code = pytest.main(args=self.args)
        if code == 1:
            cleaned_msg = ''
            traceback_str = '\n'.join(traceback)
            if self.strip_traceback_to_err:
                first_err_msg_re = (
                    r'^E\s+.*?(\w+Error|\w+Exception):([\w\W]+?)(?:^.+\1)')
                found = re.findall(
                    first_err_msg_re, traceback_str, re.MULTILINE)
                if found:
                    # last error in traceback is AssertionError or uncaught
                    # exception
                    cleaned_msg = self.clean_msg(found[-1][1])
            if cleaned_msg:
                assert False, cleaned_msg


class DiffGetter:
    """Returns added/removed lines from
    ndiff = difflib.ndiff(`diff_from`, ...) output,
    with indexes indicating the lines' position in the `diff_from` list where
    they were inserted or removed from.

    Two iterators are created to traverse `diff_from` and `ndiff` and advanced
    simultaneously. The `ndiff_iter` iterator is abreast of `diff_from_iter`
    or ahead of it, since `ndiff` is at least as large as `diff_from`.

    The `ndiff_iter`'s head points at the position in precode where change
    took place, indicated by `for_diff_code`.

    Negative difference code '- ' indicates the line was present in the
    `diff_from`, but missing from `ndiff`, which means we always advance
    ``diff_from_iter` for this code.
    """

    @dataclass
    class State:
        diff_code: str  # diff code of line from self.ndiff
        diff_line_no_code: str  # line stripped off code in self.ndiff
        diff_from_line_ix: int  # line index in self.diff_from
        diff_from_line: str  # line from self.diff_from (for debug)

    def __init__(self, ndiff: list, diff_from: list, for_diff_code='+ '):

        self.ndiff = self.sort_ndiff_items(ndiff)
        self.ndiff_iter = iter(self.ndiff)
        self.diff_from_iter = iter(diff_from)

        self.diff_code_del = '- '
        self.diff_code_add = '+ '
        self.diff_code_eql = '  '
        self.diff_code_inf = '? '

        self.for_diff_code = for_diff_code
        self.assert_diff_code()

        if for_diff_code == self.diff_code_del:
            self.skip_codes = ({self.diff_code_add, self.diff_code_inf})
        elif for_diff_code == self.diff_code_add:
            self.skip_codes = ({self.diff_code_inf})
        else:
            assert False

        diff_code = self.diff_code_eql  # will advance both iterators at start
        self.state = self.State(
            diff_code=diff_code, diff_line_no_code='',
            diff_from_line_ix=0, diff_from_line='')

    @staticmethod
    def sort_ndiff_items(ndiff_items):
        # sort items within blocks with changes based
        # - on `+`/`-` marker, with marker `?` stuck to its preceding line
        # - on the position of marked line within block
        order = []
        block_ix = 0
        prev_marker: Optional[str] = None
        marker_order = {
            '': 0,
            '-': 1,
            '+': 2,
        }
        marker_ix = 0
        for line_ix, item in enumerate(ndiff_items):
            marker = item[:2].strip()
            if bool(marker) != bool(prev_marker):
                block_ix += 1
            if marker != '?':
                marker_ix = marker_order[marker]
            order.append((block_ix, marker_ix, line_ix))
            prev_marker = marker
        return [item for _, item in sorted(zip(order, ndiff_items),
                                           key=lambda x: x[0])]

    def assert_diff_code(self):
        assert self.for_diff_code in (
            self.diff_code_del, self.diff_code_add), (
            f'`for_diff_code` expected in '
            f'({self.diff_code_del}, {self.diff_code_add})'
        )

    def get_diff(self, only_1st_diffs=False) -> Tuple[List[str], List[int]]:
        """
        See class docs for description

        :param only_1st_diffs: for several consecutive lines that differ only
            includes the 1st one in the return value
        :returns: a tuple with list of changed lines/blocks of text and
            their positions  in the `self.diff_from` list.
        """
        result_dict = defaultdict(list)
        while True:
            try:
                if self.state.diff_code == self.for_diff_code:
                    if self.for_diff_code == self.diff_code_add:
                        # no need to advance self.diff_from_iter on each
                        # addition, fast forwarding self.ndiff_iter
                        while self.state.diff_code == self.for_diff_code:
                            result_dict[self.state.diff_from_line_ix].append(
                                self.state.diff_line_no_code)
                            self.advance_ndiff()
                    else:
                        # for each deletion a line is present both in
                        # self.diff_from and self.ndiff, so we need to advance
                        # corresponding iterators simultaneously
                        result_dict[self.state.diff_from_line_ix].append(
                            self.state.diff_line_no_code)
                        self.advance_ndiff()
                        self.advance_diff_from()
                elif self.state.diff_code in self.skip_codes:
                    self.advance_ndiff()
                elif self.state.diff_code == self.diff_code_eql or (
                        self.state.diff_code == self.diff_code_del):
                    self.advance_ndiff()
                    self.advance_diff_from()
                else:
                    assert False

            except StopIteration:
                break
        return self.rehash_ix_diff_dict(result_dict, only_1st_diffs)

    def advance_ndiff(self):
        diff_line_ = ''
        while not diff_line_:
            diff_line_ = next(self.ndiff_iter)
        self.state.diff_code = diff_line_[:2]
        self.state.diff_line_no_code = diff_line_[2:]

    def advance_diff_from(self):
        self.state.diff_from_line = next(self.diff_from_iter)
        self.state.diff_from_line_ix += 1

    @staticmethod
    def rehash_ix_diff_dict(ix_diff_dict: Dict[int, list], only_1st_diffs: bool
                            ) -> Tuple[List[str], List[int]]:
        """
        Transforms dict
        {1: ['line1.1', 'line1.2'], 2: ['line2']}
        into to
        ('line1.1\nline1.2', 'line2'), (1, 2)"""

        def join_lines(lines: list) -> Optional[str]:
            nonlocal only_1st_diffs
            if not lines:
                return
            if only_1st_diffs:
                return lines[0]
            else:
                return '\n'.join(lines)

        unpack = list(
            zip(*[(join_lines(v), k) for k, v in ix_diff_dict.items()]))
        if not unpack:
            unpack = [[], []]
        diff_lines, diff_line_ixs = unpack
        return diff_lines, diff_line_ixs


DECLINE_DICT_FLAT = {'блок|ablt_plur': 'блоками', 'блок|ablt_sing': 'блоком',
                     'блок|accs_plur': 'блоки', 'блок|accs_sing': 'блок',
                     'блок|datv_plur': 'блокам', 'блок|datv_sing': 'блоку',
                     'блок|gent_plur': 'блоков', 'блок|gent_sing': 'блока',
                     'блок|loct_plur': 'блоках', 'блок|loct_sing': 'блоке',
                     'блок|nomn_plur': 'блоки', 'блок|nomn_sing': 'блок',
                     'добавленный|ablt_femn_sing': 'добавленной',
                     'добавленный|ablt_femn_sing_V_oy': 'добавленною',
                     'добавленный|ablt_masc_sing': 'добавленным',
                     'добавленный|ablt_neut_sing': 'добавленным',
                     'добавленный|ablt_plur': 'добавленными',
                     'добавленный|accs_anim_masc_sing': 'добавленного',
                     'добавленный|accs_anim_plur': 'добавленных',
                     'добавленный|accs_femn_sing': 'добавленную',
                     'добавленный|accs_inan_masc_sing': 'добавленный',
                     'добавленный|accs_inan_plur': 'добавленные',
                     'добавленный|accs_neut_sing': 'добавленное',
                     'добавленный|datv_femn_sing': 'добавленной',
                     'добавленный|datv_masc_sing': 'добавленному',
                     'добавленный|datv_neut_sing': 'добавленному',
                     'добавленный|datv_plur': 'добавленным',
                     'добавленный|femn_gent_sing': 'добавленной',
                     'добавленный|femn_loct_sing': 'добавленной',
                     'добавленный|femn_nomn_sing': 'добавленная',
                     'добавленный|gent_masc_sing': 'добавленного',
                     'добавленный|gent_neut_sing': 'добавленного',
                     'добавленный|gent_plur': 'добавленных',
                     'добавленный|loct_masc_sing': 'добавленном',
                     'добавленный|loct_neut_sing': 'добавленном',
                     'добавленный|loct_plur': 'добавленных',
                     'добавленный|masc_nomn_sing': 'добавленный',
                     'добавленный|neut_nomn_sing': 'добавленное',
                     'добавленный|nomn_plur': 'добавленные',
                     'значение|ablt_plur': 'значениями',
                     'значение|ablt_plur_V_be': 'значеньями',
                     'значение|ablt_sing': 'значением',
                     'значение|ablt_sing_V_be': 'значеньем',
                     'значение|accs_plur': 'значения',
                     'значение|accs_plur_V_be': 'значенья',
                     'значение|accs_sing': 'значение',
                     'значение|accs_sing_V_be': 'значенье',
                     'значение|datv_plur': 'значениям',
                     'значение|datv_plur_V_be': 'значеньям',
                     'значение|datv_sing': 'значению',
                     'значение|datv_sing_V_be': 'значенью',
                     'значение|gent_plur': 'значений',
                     'значение|gent_sing': 'значения',
                     'значение|gent_sing_V_be': 'значенья',
                     'значение|loct_plur': 'значениях',
                     'значение|loct_plur_V_be': 'значеньях',
                     'значение|loct_sing': 'значении',
                     'значение|loct_sing_V_be': 'значенье',
                     'значение|loct_sing_V_be_V_bi': 'значеньи',
                     'значение|nomn_plur': 'значения',
                     'значение|nomn_plur_V_be': 'значенья',
                     'значение|nomn_sing': 'значение',
                     'значение|nomn_sing_V_be': 'значенье',
                     'изменённый|ablt_femn_sing': 'изменённой',
                     'изменённый|ablt_femn_sing_V_oy': 'изменённою',
                     'изменённый|ablt_masc_sing': 'изменённым',
                     'изменённый|ablt_neut_sing': 'изменённым',
                     'изменённый|ablt_plur': 'изменёнными',
                     'изменённый|accs_anim_masc_sing': 'изменённого',
                     'изменённый|accs_anim_plur': 'изменённых',
                     'изменённый|accs_femn_sing': 'изменённую',
                     'изменённый|accs_inan_masc_sing': 'изменённый',
                     'изменённый|accs_inan_plur': 'изменённые',
                     'изменённый|accs_neut_sing': 'изменённое',
                     'изменённый|datv_femn_sing': 'изменённой',
                     'изменённый|datv_masc_sing': 'изменённому',
                     'изменённый|datv_neut_sing': 'изменённому',
                     'изменённый|datv_plur': 'изменённым',
                     'изменённый|femn_gent_sing': 'изменённой',
                     'изменённый|femn_loct_sing': 'изменённой',
                     'изменённый|femn_nomn_sing': 'изменённая',
                     'изменённый|gent_masc_sing': 'изменённого',
                     'изменённый|gent_neut_sing': 'изменённого',
                     'изменённый|gent_plur': 'изменённых',
                     'изменённый|loct_masc_sing': 'изменённом',
                     'изменённый|loct_neut_sing': 'изменённом',
                     'изменённый|loct_plur': 'изменённых',
                     'изменённый|masc_nomn_sing': 'изменённый',
                     'изменённый|neut_nomn_sing': 'изменённое',
                     'изменённый|nomn_plur': 'изменённые',
                     'имя|Abbr_gent_sing': 'им', 'имя|ablt_plur': 'именами',
                     'имя|ablt_sing': 'именем', 'имя|accs_plur': 'имена',
                     'имя|accs_sing': 'имя', 'имя|datv_plur': 'именам',
                     'имя|datv_sing': 'имени', 'имя|gent_plur': 'имён',
                     'имя|gent_sing': 'имени', 'имя|loct_plur': 'именах',
                     'имя|loct_sing': 'имени', 'имя|nomn_plur': 'имена',
                     'имя|nomn_sing': 'имя',
                     'переменный|ablt_femn_sing': 'переменной',
                     'переменный|ablt_femn_sing_V_oy': 'переменною',
                     'переменный|ablt_masc_sing': 'переменным',
                     'переменный|ablt_neut_sing': 'переменным',
                     'переменный|ablt_plur': 'переменными',
                     'переменный|accs_anim_masc_sing': 'переменного',
                     'переменный|accs_anim_plur': 'переменных',
                     'переменный|accs_femn_sing': 'переменную',
                     'переменный|accs_inan_masc_sing': 'переменный',
                     'переменный|accs_inan_plur': 'переменные',
                     'переменный|accs_neut_sing': 'переменное',
                     'переменный|datv_femn_sing': 'переменной',
                     'переменный|datv_masc_sing': 'переменному',
                     'переменный|datv_neut_sing': 'переменному',
                     'переменный|datv_plur': 'переменным',
                     'переменный|femn_gent_sing': 'переменной',
                     'переменный|femn_loct_sing': 'переменной',
                     'переменный|femn_nomn_sing': 'переменная',
                     'переменный|gent_masc_sing': 'переменного',
                     'переменный|gent_neut_sing': 'переменного',
                     'переменный|femn_gent_plur': 'переменных',
                     'переменный|gent_plur': 'переменных',
                     'переменный|loct_masc_sing': 'переменном',
                     'переменный|loct_neut_sing': 'переменном',
                     'переменный|loct_plur': 'переменных',
                     'переменный|masc_nomn_sing': 'переменный',
                     'переменный|neut_nomn_sing': 'переменное',
                     'переменный|nomn_plur': 'переменные',
                     'путь|ablt_plur': 'путями', 'путь|ablt_sing': 'путём',
                     'путь|accs_plur': 'пути', 'путь|accs_sing': 'путь',
                     'путь|datv_plur': 'путям', 'путь|datv_sing': 'пути',
                     'путь|gent_plur': 'путей', 'путь|gent_sing': 'пути',
                     'путь|loct_plur': 'путях', 'путь|loct_sing': 'пути',
                     'путь|nomn_plur': 'пути', 'путь|nomn_sing': 'путь',
                     'строка|ablt_plur': 'строками',
                     'строка|ablt_sing': 'строкой',
                     'строка|ablt_sing_V_oy': 'строкою',
                     'строка|accs_plur': 'строки',
                     'строка|accs_sing': 'строку',
                     'строка|datv_plur': 'строкам',
                     'строка|datv_sing': 'строке', 'строка|gent_plur': 'строк',
                     'строка|gent_sing': 'строки',
                     'строка|loct_plur': 'строках',
                     'строка|loct_sing': 'строке',
                     'строка|nomn_plur': 'строки',
                     'строка|nomn_sing': 'строка',
                     'текст|ablt_plur': 'текстами',
                     'текст|ablt_sing': 'текстом', 'текст|accs_plur': 'тексты',
                     'текст|accs_sing': 'текст', 'текст|datv_plur': 'текстам',
                     'текст|datv_sing': 'тексту', 'текст|gent_plur': 'текстов',
                     'текст|gent_sing': 'текста', 'текст|loct_plur': 'текстах',
                     'текст|loct_sing': 'тексте', 'текст|nomn_plur': 'тексты',
                     'текст|nomn_sing': 'текст',
                     'удалённый|ablt_femn_sing': 'удалённой',
                     'удалённый|ablt_femn_sing_V_oy': 'удалённою',
                     'удалённый|ablt_masc_sing': 'удалённым',
                     'удалённый|ablt_neut_sing': 'удалённым',
                     'удалённый|ablt_plur': 'удалёнными',
                     'удалённый|accs_anim_masc_sing': 'удалённого',
                     'удалённый|accs_anim_plur': 'удалённых',
                     'удалённый|accs_femn_sing': 'удалённую',
                     'удалённый|accs_inan_masc_sing': 'удалённый',
                     'удалённый|accs_inan_plur': 'удалённые',
                     'удалённый|accs_neut_sing': 'удалённое',
                     'удалённый|datv_femn_sing': 'удалённой',
                     'удалённый|datv_masc_sing': 'удалённому',
                     'удалённый|datv_neut_sing': 'удалённому',
                     'удалённый|datv_plur': 'удалённым',
                     'удалённый|femn_gent_sing': 'удалённой',
                     'удалённый|femn_loct_sing': 'удалённой',
                     'удалённый|femn_nomn_sing': 'удалённая',
                     'удалённый|gent_masc_sing': 'удалённого',
                     'удалённый|gent_neut_sing': 'удалённого',
                     'удалённый|gent_plur': 'удалённых',
                     'удалённый|loct_masc_sing': 'удалённом',
                     'удалённый|loct_neut_sing': 'удалённом',
                     'удалённый|loct_plur': 'удалённых',
                     'удалённый|masc_nomn_sing': 'удалённый',
                     'удалённый|neut_nomn_sing': 'удалённое',
                     'удалённый|nomn_plur': 'удалённые'}


class CompareLinesDiffWithAuthor(CompareDiffWithAuthor):
    """
    Checks that:
    - number of changed items of precode in author's and student's solutions
    are the same,
    - there is no difference between author and student solutions.
    """

    @property
    def checked_item_name(self):
        """The name of the kind of items being compared for use in
        except messages"""
        return 'код'

    @property
    def add_or_edit_item_msg(self):
        """
        Message to show to user.
        Text in curly brackets will be replaced when message is rendered.
        In cases like {option1/option2} only one of the options will
        remain.
        """
        return (
            'Убедитесь, что код \n'
            '`{add_item}` \n'
            'есть в решении, не перемещён, не дублируется '
            'и что в нём нет ошибок.\n'
            '{how_fix}'
        )

    @property
    def how_fix_msg(self):
        """
        Message to show to user.
        Text in curly brackets will be replaced when message is rendered.
        In cases like {option1/option2} only one of the options will
        remain.
        """
        if self.student_item:
            return (
                'Попробуйте вставить код перед, после или вместо кода\n '
                '\n'
                '`{student_item}`\n'
                '\n'
                'В последнем случае в него '
                'необходимо внести следующие изменения '
                'после строки прекода №{diffs_to_make_line_pos} '
                '(строки комментариев не учитываются): '
                '\n'
                '`\n{diffs_to_make_text}\n`'
                '(где `-` - удалить, `+` - добавить, `?` - позиция различий).'
            )
        else:
            return (
                'Изменения необходимо внести '
                'после строки прекода №{diffs_to_make_line_pos} '
                '(строки комментариев не учитываются).'
            )

    @property
    def match_n_items_msg(self):
        """
        Message to show to user.
        Text in curly brackets will be replaced when message is rendered.
        In cases like {option1/option2} only one of the options will
        remain.
        """
        return (
            'Убедитесь, что количество участков кода, '
            '{добавленных/удалённых}/изменённых в прекоде файла\n '
            '`{student_file}`\n'
            ', равно {operated_n_author} (строки комментариев не '
            'учитываются).\n '
            'В настоящий момент число {добавленных/удалённых}/изменённых '
            'участков кода в прекоде равно {operated_n_student}:\n '
            '`\n'
            '{diff}\n'
            '`'
        )

    @property
    def remove_or_edit_item_msg(self):
        """
        Message to show to user.
        Text in curly brackets will be replaced when message is rendered.
        In cases like {option1/option2} only one of the options will
        remain.
        """
        return (
            'Код\n'
            '`{remove_item}`\n'
            'необходимо изменить или удалить '
            'и убедиться, что он не дублируется.\n '
            '{how_fix}'
        )

    def make_items(self, student_items: list, author_items: list,
                   precode_items: list, strip_items=False):
        """Splits precode and solutions into lines storing them as items
        for further comparison"""
        self.student_items = student_items
        self.author_items = author_items
        self.precode_items = precode_items
        if strip_items:
            self.student_items = [s.strip() for s in self.student_items]
            self.author_items = [a.strip() for a in self.author_items]
            self.precode_items = [p.strip() for p in self.precode_items]


class CapturingStdout(list):
    """
    Context manager for capturing stdout.
    Usage:
        with Capturing() as func_output:
            func()
    check func() output in func_output variable
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        self.append(self._stringio.getvalue())
        self._stringio.close()
        del self._stringio


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
        assert False, f'Ошибка в файле `{student_path.relative_to(student_path.parent.parent.parent)}`:\n{str(e)} '


if __name__ == '__main__':
    run_pytest_runner(__file__)
