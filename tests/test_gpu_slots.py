"""Rationing the GPU between the Docling workers and the embedding model.

They share one device and neither yields, so without a limit three converting
workers plus an embedding batch exhausted a 23 GB L4 three different ways in
one afternoon.
"""

import multiprocessing as mp
import threading
import time

import pytest

import mcp_manual_walker.builder as builder


@pytest.fixture
def slot(monkeypatch):
    def install(count):
        sem = mp.get_context("spawn").Semaphore(count) if count else None
        monkeypatch.setattr(builder, "_gpu_slot", sem)
        return sem

    return install


def test_without_a_semaphore_the_block_just_runs(slot):
    slot(0)
    with builder.gpu_slot("nothing configured"):
        pass


def test_the_slot_is_held_for_the_block(slot):
    sem = slot(1)
    with builder.gpu_slot("work"):
        assert sem.acquire(block=False) is False
    assert sem.acquire(block=False) is True
    sem.release()


def test_the_slot_is_released_even_when_the_block_raises(slot):
    # A conversion that dies must not take a slot with it, or the build
    # deadlocks one worker at a time until nothing can run.
    sem = slot(1)
    with pytest.raises(RuntimeError):
        with builder.gpu_slot("doomed"):
            raise RuntimeError("boom")
    assert sem.acquire(block=False) is True
    sem.release()


def test_only_as_many_run_at_once_as_there_are_slots(slot):
    slot(2)
    concurrent, peak, lock = 0, 0, threading.Lock()

    def worker():
        nonlocal concurrent, peak
        with builder.gpu_slot("work"):
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.15)
            with lock:
                concurrent -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 2


def test_the_embedder_competes_for_the_same_slots(slot):
    """The point of the design: embedding costs a converting worker.

    Three slots and three busy workers means the parent waits, and while it
    holds one only two workers convert.
    """
    slot(3)
    held = []

    def convert(name):
        with builder.gpu_slot(name):
            held.append(name)
            time.sleep(0.2)
            held.remove(name)

    workers = [threading.Thread(target=convert, args=(f"w{i}",)) for i in range(3)]
    for t in workers:
        t.start()
    time.sleep(0.05)
    assert len(held) == 3  # all three slots taken by conversions

    embedded = threading.Event()

    def embed():
        with builder.gpu_slot("embedding"):
            embedded.set()

    parent = threading.Thread(target=embed)
    parent.start()
    assert not embedded.wait(timeout=0.05)  # blocked while the workers hold them
    for t in workers:
        t.join()
    parent.join()
    assert embedded.is_set()


def test_a_slot_is_not_leaked_across_uses(slot):
    sem = slot(1)
    for _ in range(5):
        with builder.gpu_slot("work"):
            pass
    assert sem.acquire(block=False) is True
    sem.release()
