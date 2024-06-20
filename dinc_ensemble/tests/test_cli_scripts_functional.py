import os
import pytest

def entrypoint_test(cli_path: str, command: str) -> int:
    #if this is not in a docker container it should be just python
    return os.system("/opt/conda/envs/DINC-Ensemble/bin/python {} {} --help".format(cli_path, command))

def test_entrypoint_analysis(cli_main):
    ret_val = entrypoint_test(cli_main, "analyze")
    assert ret_val == 0


def test_params_analysis():
    pass

def test_entrypoint_fragmentation(cli_main: str):
    ret_val = entrypoint_test(cli_main, "fragment")
    assert ret_val == 0


def test_params_fragmentation():
    pass

def test_entrypoint_core(cli_main: str):
    ret_val = entrypoint_test(cli_main, "dock")
    assert ret_val == 0


def test_params_core():
    pass

