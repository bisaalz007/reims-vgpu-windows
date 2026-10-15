"""Loopback QMP protocol fixture. No VM, guest image or hypervisor is used."""
import argparse
import json
import pathlib
import socketserver
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--ready', type=pathlib.Path, required=True)
options = parser.parse_args()


class QmpHandler(socketserver.StreamRequestHandler):
    def send(self, value):
        self.wfile.write((json.dumps(value) + '\n').encode())
        self.wfile.flush()

    def handle(self):
        self.send({'QMP': {'version': {}, 'capabilities': []}})
        for line in self.rfile:
            request = json.loads(line)
            command = request['execute']
            self.send({'event': 'FIXTURE_EVENT'})
            self.send({'return': {'unrelated': True}, 'id': 'another-request'})
            if command == 'fixture-error':
                self.send({'error': {'class': 'GenericError', 'desc': 'fixture failure'}, 'id': request['id']})
                continue
            result = {}
            if command == 'query-status':
                result = {'running': self.server.state == 'running', 'status': self.server.state}
            elif command == 'query-block':
                result = []
            elif command == 'qom-list':
                result = [{'name': key} for key in self.server.usb]
            elif command == 'device_add':
                self.server.usb.add(request['arguments']['id'])
            elif command == 'device_del':
                self.server.usb.remove(request['arguments']['id'])
            elif command == 'fixture-echo':
                result = request['arguments']
            elif command == 'screendump':
                pathlib.Path(request['arguments']['filename']).write_bytes(b'P6\n1 1\n255\n\x01\x02\x03')
            elif command == 'system_powerdown':
                if self.server.state == 'running':
                    self.server.state = 'shutdown'
            elif command == 'system_reset':
                if self.server.state != 'running':
                    self.server.state = 'prelaunch'
            elif command == 'stop':
                self.server.state = 'paused'
            elif command == 'cont':
                if self.server.state == 'shutdown':
                    self.send({'error': {'class': 'GenericError', 'desc': 'Resetting the Virtual Machine is required'}, 'id': request['id']})
                    continue
                self.server.state = 'running'
            self.send({'return': result, 'id': request['id']})
            if command == 'quit':
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return


with socketserver.ThreadingTCPServer(('127.0.0.1', 0), QmpHandler) as server:
    server.daemon_threads = True
    server.state = 'running'
    server.usb = set()
    options.ready.write_text(json.dumps({'port': server.server_address[1]}), encoding='utf-8')
    server.serve_forever(poll_interval=0.05)
