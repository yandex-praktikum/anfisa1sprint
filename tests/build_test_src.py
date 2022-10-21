from yap_testing.manage import webpack_django_test_src


build_in = [
    '../author',
]

if __name__ == '__main__':
    webpack_django_test_src(
        django_paths=build_in,
        dry_run=False,
        move_test_src_webpacked_to='../tests/test.py',
        create_pytest_ini_file=True
    )
