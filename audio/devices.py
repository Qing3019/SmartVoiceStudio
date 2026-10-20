
import sounddevice as sd


def get_audio_devices():

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    input_devices = []
    output_devices = []

    for index, device in enumerate(devices):

        api_name = hostapis[device["hostapi"]]["name"]
        device_name = device["name"]
        sample_rate = int(device["default_samplerate"])

        label = f"[{index}] {device_name} ({api_name}) - {sample_rate} Hz"

        if device["max_input_channels"] > 0:
            input_devices.append((index, label))

        if device["max_output_channels"] > 0:
            output_devices.append((index, label))

    return input_devices, output_devices


def get_input_sample_rate(device=None):

    device_info = sd.query_devices(device, "input")
    sample_rate = int(round(device_info["default_samplerate"]))

    if sample_rate <= 0:
        raise ValueError("输入设备没有有效的默认采样率")

    sd.check_input_settings(
        device=device,
        channels=1,
        samplerate=sample_rate,
        dtype="float32"
    )

    return sample_rate
