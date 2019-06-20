import signal
import subprocess
import sys
import time
import atexit
import os


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        # logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


class GPUEnergyEvaluator(object):
    def __init__(self, subprocess_cmd=None, gpuid=0, watts_offset=False):
        watts_idle = 0.0
        if watts_offset:
            # tic = time.time()
            for _ in range(10):
                watts_idle += float(subprocess.getoutput(
                    ['nvidia-smi --id={} --format=csv,noheader --query-gpu=power.draw'.format(gpuid)])[:-1])
            watts_idle /= 10.0
            # print('watts_idle:{}, cost{}s'.format(watts_idle, time.time() - tic))
        if subprocess_cmd is None:
            self.subprocess_cmd = ['python', os.path.realpath(__file__), str(gpuid), str(watts_idle)]
        else:
            self.subprocess_cmd = subprocess_cmd
        self.p = None

    def start(self):
        self.p = subprocess.Popen(self.subprocess_cmd, stdout=subprocess.PIPE)

    def end(self):
        if self.p is not None:
            # self.p.terminate()
            self.p.send_signal(signal.SIGINT)
            # self.p.terminate()
            self.p.wait()
            try:
                energy = self.p.communicate()[0]
                # energy = self.p.stdout.readline()
            except subprocess.TimeoutExpired:
                # ignore
                pass
            assert energy[-2:] == b"J\n"
            return float(energy[:-2])


if __name__ == '__main__':
    gpuid = int(sys.argv[1])
    watts_idle = float(sys.argv[2])
    cmdline = ['nvidia-smi --id={} --format=csv,noheader --query-gpu=power.draw'.format(gpuid)]
    energy_used = 0.0
    atexit.register(lambda: print("{}J".format(energy_used)))
    int_lock = DelayedKeyboardInterrupt()
    try:
        time_a = time.time()
        while True:
            with int_lock:
                cur_watts = float(subprocess.getoutput(cmdline)[:-1])
            time.sleep(0.001)
            time_b = time.time()
            energy_used += max(cur_watts - watts_idle, 0.0) * (time_b - time_a)
            time_a = time_b
    except KeyboardInterrupt:
        # ignore
        pass
