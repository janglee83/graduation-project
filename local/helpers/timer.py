import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def elapsed_time(self, text: str):
        if self.start_time is None:
            raise ValueError("Timer has not been started yet.")
        if self.end_time is None:
            raise ValueError("Timer has not been ended yet.")

        print(text, self.end_time - self.start_time)
