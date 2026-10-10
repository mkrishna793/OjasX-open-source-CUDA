import torch


def pytest_sessionstart(session):
    torch.manual_seed(20261006)
    torch.set_num_threads(2)
