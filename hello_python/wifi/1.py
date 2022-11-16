import nmcli

wifi_list = nmcli.device.wifi()
for device_wifi in wifi_list:
    print(device_wifi, end='\n\n')
