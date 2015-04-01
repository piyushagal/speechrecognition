__author__ = 'piyush'
import pyaudio
import wave


class Recorder:

    def record(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 1
        self.p = pyaudio.PyAudio()

        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        print("Recording ....")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("Done Recording ... ")

        stream.stop_stream()
        stream.close()
        self.p.terminate()

        return frames

    def save(self, frames, fileName):
        wf = wave.open(fileName, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()



