class Logger:
    def __init__(self, filename):
        self.filename = filename

    def write_line(self, message):
        with open(self.filename, "a") as file:
            file.write(message + "\n")