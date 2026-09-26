import sounddevice as sd

print("所有音频设备：")
print(sd.query_devices())

print("\n默认设备：")
print(sd.default.device)