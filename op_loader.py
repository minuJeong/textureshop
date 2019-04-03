
from op_base import Base


class ImageLoader(Base):

    def in_node(self, in_node):
        self.W, self.H = in_node.W, in_node.H

        return self

    def out_node(self):
        pass
