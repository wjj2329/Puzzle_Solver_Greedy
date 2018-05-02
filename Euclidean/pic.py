import uuid
class Pic:
    def __init__(self, pic, name):
      self.pic = pic
      self.name=name
      self.UUID=uuid.uuid4()
