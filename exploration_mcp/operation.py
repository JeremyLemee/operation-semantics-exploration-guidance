from pydoc import describe


class Operation:
    def __init__(self):
        self._description = None
        self._name = None

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    def __str__(self):
        s = "The operation has name "
        s += self._name
        s += " and description "
        s += self._description
        return s
