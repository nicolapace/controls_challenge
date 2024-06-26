from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    A simple PID controller
    """

    def __init__(
        self,
    ):
        self.p = 0.0404
        self.i = 0.1041
        self.d = 0.0232
        # Best individual: p=0.0737, i=0.1242, d=0.0580

        self.p = 0.0737
        self.i = 0.1242
        self.d = 0.0280

        self.p = 0.0888
        self.i = 0.1302
        self.d = 0.0082

        # self.p = 0.1073
        # self.i = 0.0801
        # self.d = 0.0098
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff
