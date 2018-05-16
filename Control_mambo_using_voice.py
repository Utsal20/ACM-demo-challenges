r"""
Controlling a drone using Voice Commands
@ Authors: Abdul Haq, Utsal Shrestha
@ Reference: Tensorflow audio recognition
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

from Mambo import Mambo
import sys

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None

import keyboard
import pyaudio
import wave

# settings for wav files
CHUNK = 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "/home/pirate/tensorflow-master/tensorflow/examples/speech_commands/output1.wav"

def record_audio():
	"""
	This function takes a 1s stream of audio and saves it as a wav file
	"""
    p = pyaudio.PyAudio()
	print("Recording...", end='')
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recorded")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def exec_command(cm):
	"""
	This function uses conditional branching statements to execute appropriate commands given a command string
	@params:
		cm -> command string
	"""
    if cm == '_silence_' or cm == '_unknown_' or cm == 'yes' or cm == 'no':
        return
    elif cm == 'on':
        mambo.safe_takeoff(5)
    elif cm == 'off':
        mambo.safe_land(5)
        mambo.smart_sleep(5)
    elif cm == 'up':
        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=80, duration=1)
    elif cm == 'down':
        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-50, duration=1)
    elif cm == 'left':
        mambo.fly_direct(roll=0, pitch=0, yaw=-90, vertical_movement=0, duration=1)
    elif cm == 'right':
        mambo.fly_direct(roll=0, pitch=0, yaw=90, vertical_movement=0, duration=1)
    elif cm == 'go':
        mambo.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration=1)
    elif cm == 'stop':
        mambo.fly_direct(roll=0, pitch=-80, yaw=0, vertical_movement=0, duration=1)

def load_graph(filename):
  """
  Unpersists graph from file as default graph.
  """
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """
  Read in labels, one label per line.
  """
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph():
  """
  Runs the audio data through the graph and prints predictions.
  """
  # load labels
  labels = load_labels('/home/pirate/Downloads/speech_commands/conv_actions_labels.txt')

  # load graph, which is stored in the default session
  load_graph('/home/pirate/Downloads/speech_commands/conv_actions_frozen.pb')

  wav = '/home/pirate/tensorflow-master/tensorflow/examples/speech_commands/output1.wav'

  with open(wav, 'rb') as wav_file:
      wav_data = wav_file.read()

  # input and output files
  input_layer_name = 'wav_data:0'
  output_layer_name = 'labels_softmax:0'

  num_top_predictions = 3

  # the code below is from a file called label_wav.py in the tensorflow source tree
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    print(labels[top_k[0]])
    for i in top_k:
        print(labels[i])

	#returns the string with the highest probability
    return labels[top_k[0]]

# drone setup
mamboAddr = ""
mambo = Mambo(mamboAddr, use_wifi=True)
print("Connecting...")
success = mambo.connect(num_retries=3)
print("Connected: %s" %success)

if (not success):
    sys.exit()

mambo.smart_sleep(2)
mambo.ask_for_state_update()
mambo.smart_sleep(2)
mambo.safe_takeoff(5)

print('Start.')
while True:
  # checking keyboard button press events
  if keyboard.is_pressed('space'):
      record_audio()
      cm = run_graph()
      exec_command(cm)
  if keyboard.is_pressed('q'):
      print("Quit.")
      mambo.disconnect()
      break