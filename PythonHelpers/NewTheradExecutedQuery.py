import threading, queue
from time import sleep
from typing import Callable, Any


class NewThreadExecutedQueue:
    def __init__(self, handler: Callable[[Any], None]):
        self.__queue__ = queue.Queue()
        self.__handler__ = handler
        self.__is_closed__ = False
        self.__thread__ = None

    def enqueue(self, item: Any):
        assert not self.__is_closed__
        self.__queue__.put(item)

    def start(self):
        self.__thread__ = threading.Thread(target=self.__execute__)
        self.__thread__.start()

    def close(self):
        assert not self.__is_closed__
        self.enqueue(None)
        self.__is_closed__ = True

    def __execute__(self):
        just_started = True
        item = 'stub'
        while item is not None:
            if not just_started:
                self.__handler__(item)
            else:
                just_started = False
            while self.__queue__.empty():
                sleep(5)
            item = self.__queue__.get()
        self.__thread__.join()

