import socket
import json

def send_request(ip, command):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, 4028))
    sock.sendall(json.dumps({"command":command}).encode())
    data = sock.recv(4096)
    sock.close()
    return json.loads(data.decode())

# Replace with the IP address of your Antminer
antminer_ip = '192.168.1.100'

# Send a request to get the miner status
response = send_request(antminer_ip, 'stats')

# Print the temperatures
print('Temperature 1: ', response['STATS'][1]['temp1'])
print('Temperature 2: ', response['STATS'][1]['temp2'])
print('Temperature 3: ', response['STATS'][1]['temp3'])