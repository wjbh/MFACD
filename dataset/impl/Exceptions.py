class KnownException(Exception):
    def __init__(self, msg):
        super(KnownException, self).__init__(msg)

class DecodeErrorException(Exception):
    def __init__(self, msg):
        super(DecodeErrorException, self).__init__(msg)