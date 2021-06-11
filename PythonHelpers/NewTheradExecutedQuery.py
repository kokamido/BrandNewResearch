import threading, queue
from typing import Callable, Any


class NewThreadExecutedQueue:
    def __init__(self, handler: Callable[[Any], None]):
        self.__queue__ = queue.Queue()
        self.__handler__ = handler
        self.__is_closed__ = False

    def enqueue(self, item: Any):
        assert item
        assert not self.__is_closed__
        self.__queue__.put(item)

    def start(self):
        threading.Thread(target=self.__execute__).start()

    def close(self):
        assert not self.__is_closed__
        self.__is_closed__ = True
        self.enqueue(None)

    def __execute__(self):
        for item in iter(self.__queue__.get, None):
            if item is not None:
                self.__handler__(item)
