"""Native AVFoundation capture, imported only inside the macOS camera worker."""

import math
import time
from queue import Empty, Full, Queue
from threading import Event

import AVFoundation as av
import CoreMedia as cm
import cv2
import dispatch
import numpy as np
import objc
import Quartz as cv
from Foundation import NSObject

from .models import CameraSpec


def _format(device, spec: CameraSpec):
    modes = []
    for mode in device.formats():
        description = mode.formatDescription()
        if tuple(cm.CMVideoFormatDescriptionGetDimensions(description)) != (
            spec.width,
            spec.height,
        ):
            continue
        subtype = cm.CMFormatDescriptionGetMediaSubType(description)
        for rate in mode.videoSupportedFrameRateRanges():
            if rate.minFrameRate() - 0.05 <= spec.capture_fps <= rate.maxFrameRate() + 0.05:
                duration = cm.CMTimeMake(1, spec.capture_fps)
                if spec.capture_fps >= rate.maxFrameRate():
                    duration = rate.minFrameDuration()
                elif spec.capture_fps <= rate.minFrameRate():
                    duration = rate.maxFrameDuration()
                modes.append((subtype, mode, duration))
    if not modes:
        raise RuntimeError(
            f"{device.localizedName()} does not expose a native "
            f"{spec.width} × {spec.height} mode at {spec.capture_fps} FPS."
        )
    return min(
        modes,
        key=lambda mode: (
            mode[0]
            not in (
                cv.kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                cv.kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            )
        ),
    )


def _clock_offset(host_clock):
    probes = []
    for _ in range(8):
        before = time.monotonic()
        host_time = cm.CMTimeGetSeconds(cm.CMClockGetTime(host_clock))
        after = time.monotonic()
        probes.append((after - before, (before + after) / 2 - host_time))
    return min(probes)[1]


def _rgb(sample, shape):
    buffer = cm.CMSampleBufferGetImageBuffer(sample)
    if (cv.CVPixelBufferGetHeight(buffer), cv.CVPixelBufferGetWidth(buffer)) != shape:
        raise RuntimeError("The camera did not deliver the requested native resolution.")
    lock = cv.kCVPixelBufferLock_ReadOnly
    if cv.CVPixelBufferLockBaseAddress(buffer, lock):
        raise RuntimeError("Could not read the camera's pixel buffer.")
    try:
        height, width = shape
        stride = cv.CVPixelBufferGetBytesPerRow(buffer)
        data = cv.CVPixelBufferGetBaseAddress(buffer).as_buffer(stride * height)
        pixels = np.ndarray((height, width, 4), np.uint8, buffer=data, strides=(stride, 4, 1))
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
    finally:
        cv.CVPixelBufferUnlockBaseAddress(buffer, lock)


class GhostPathVideoDelegate(
    NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]
):
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        arrival = time.monotonic()
        try:
            clock = self.session.synchronizationClock()
            if clock is None:
                return
            pts = cm.CMSampleBufferGetPresentationTimeStamp(sample)
            host_pts = cm.CMSyncConvertTime(pts, clock, self.host_clock)
            stamp = cm.CMTimeGetSeconds(host_pts) + self.offset
            if not math.isfinite(stamp) or stamp <= self.last_stamp:
                return
            self.last_stamp = stamp
            item = (stamp, arrival, _rgb(sample, self.shape))
        except Exception as exc:
            item = exc
        try:
            self.frames.put_nowait(item)
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            self.frames.put_nowait(item)


def _authorize():
    status = av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeVideo)
    if status == av.AVAuthorizationStatusAuthorized:
        return
    if status == av.AVAuthorizationStatusNotDetermined:
        allowed = Event()
        av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            av.AVMediaTypeVideo, lambda granted: allowed.set() if granted else None
        )
        if allowed.wait(5):
            return
    raise RuntimeError("Allow camera access in System Settings → Privacy & Security → Camera.")


def capture(device_id: str, spec: CameraSpec, publish_mode):
    _authorize()
    cv2.setNumThreads(1)
    device = av.AVCaptureDevice.deviceWithUniqueID_(device_id)
    if device is None:
        raise RuntimeError("The selected camera is disconnected. Check its USB cable.")
    subtype, mode, duration = _format(device, spec)
    session = av.AVCaptureSession.alloc().init()
    input_device, error = av.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
    if input_device is None:
        raise RuntimeError(
            f"Could not open {device.localizedName()}: {error.localizedDescription()}"
        )
    output = av.AVCaptureVideoDataOutput.alloc().init()
    output.setAutomaticallyConfiguresOutputBufferDimensions_(False)
    output.setVideoSettings_({cv.kCVPixelBufferPixelFormatTypeKey: cv.kCVPixelFormatType_32BGRA})
    output.setAlwaysDiscardsLateVideoFrames_(True)

    session.beginConfiguration()
    if not session.canAddInput_(input_device) or not session.canAddOutput_(output):
        raise RuntimeError(f"Could not set up the feed from {device.localizedName()}.")
    session.addInput_(input_device)
    session.addOutput_(output)
    if session.canSetSessionPreset_(av.AVCaptureSessionPresetInputPriority):
        session.setSessionPreset_(av.AVCaptureSessionPresetInputPriority)

    delegate = GhostPathVideoDelegate.alloc().init()
    delegate.frames = Queue(maxsize=1)
    delegate.session = session
    delegate.host_clock = cm.CMClockGetHostTimeClock()
    delegate.offset = _clock_offset(delegate.host_clock)
    delegate.last_stamp = -math.inf
    delegate.shape = (spec.height, spec.width)
    queue = dispatch.dispatch_queue_create(b"ghostpath.camera", None)
    output.setSampleBufferDelegate_queue_(delegate, queue)

    locked, error = device.lockForConfiguration_(None)
    if not locked:
        raise RuntimeError(
            f"Could not configure {device.localizedName()}: {error.localizedDescription()}"
        )
    try:
        device.setActiveFormat_(mode)
        device.setActiveVideoMinFrameDuration_(duration)
        device.setActiveVideoMaxFrameDuration_(duration)
        session.commitConfiguration()
        session.startRunning()
    finally:
        device.unlockForConfiguration()
    try:
        if not session.isRunning() or device.activeFormat() != mode:
            raise RuntimeError("The camera did not start in the requested native mode.")
        publish_mode(
            {
                "device_id": device_id,
                "label": str(device.localizedName()),
                "width": spec.width,
                "height": spec.height,
                "nominal_fps": round(
                    1 / cm.CMTimeGetSeconds(device.activeVideoMinFrameDuration()), 3
                ),
                "format": int(subtype).to_bytes(4, "big").decode("ascii", errors="replace"),
                "timestamp_source": "avfoundation_pts",
            }
        )
        while True:
            try:
                item = delegate.frames.get(timeout=2)
            except Empty:
                raise RuntimeError(
                    "The camera stopped delivering frames with valid timestamps."
                ) from None
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        output.setSampleBufferDelegate_queue_(None, None)
        session.stopRunning()
