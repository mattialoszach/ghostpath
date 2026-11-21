"""Track verified torque state and apply the same policy to every robot adapter."""

from collections.abc import Callable

from .motion import Pose, SafetyFault
from .robot import Robot


class TorqueManager:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.state = robot.torque()

    def release(self):
        self.state = "unknown"
        self.robot.release()
        self.state = "off"

    def engage(self, check: Callable[[], None]) -> Pose:
        check()
        self.state = "unknown"
        pose = self.robot.arm(check)
        self.state = "on"
        return pose

    def refresh(self, *, active: bool = False):
        reported = self.robot.torque()
        if active and reported != self.state:
            raise SafetyFault("Motor torque changed unexpectedly.")
        self.state = reported

    def stop(self):
        self.state = "unknown"
        self.robot.hold()
        self.refresh()
