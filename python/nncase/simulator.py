from _nncase import Simulator as InternelSimulator


class Simulator(InternelSimulator):

    def __set_kmodel_bytes(self, kmodel_bytes: bytes):
        self.__kmodel_bytes = kmodel_bytes

    def __get_kmodel_bytes(self) -> bytes:
        return self.__kmodel_bytes

    kmodel_bytes: bytes = property(__get_kmodel_bytes, __set_kmodel_bytes)

    def load_model(self, model: bytes) -> None:
        self.kmodel_bytes = model  # keep the model data alive
        return super().load_model(self.kmodel_bytes)
