import tensorflow as tf
import numpy as np
import io
import csv
import wave
import pyaudio
import io

#print("Real-Time Audio Classfication with Tensorflow/PyAudio")

#CSV Reader helper function for YAMNet
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header
  return class_names

#Audio Constants
#Sample Rate
RATE=16000
#Length of each Window
RECORD_SECONDS = 2
#Buffer Size
CHUNKSIZE = 1024

# initialize portaudio object
p = pyaudio.PyAudio()

 #Set up YAMNet variables
interpreter = tf.lite.Interpreter('/Users/devansh/Desktop/RTAC/lite-model_yamnet_tflite_1.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

# Download the YAMNet class map to yamnet_class_map.csv
class_names = class_names_from_csv(open('/Users/devansh/Desktop/RTAC/yamnet_class_map.csv').read())

#Decide length of audio analysis loop. 
loopLength = 100 #Decide real-time loop length in seconds
loopcounter = 0

#Real-Time Audio Analysis Loop Starts Now.
#Real-Time Loop Runs for loopLength/2 iterations, since each loop runs for 2 seconds
while(loopcounter<=loopLength/2):
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    frames = [] # A python-list of chunks(numpy.ndarray)
    for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
        data = stream.read(CHUNKSIZE)
        frames.append(np.frombuffer(data, dtype=np.int16))

    #Convert the list of numpy-arrays into a 1D array (column-wise)
    numpydata = np.hstack(frames)
    #Converting numpy array from int16 to float32 and scaling accordingly
    numpydata = numpydata.astype(np.float32, order='C') / 32768.0

    # close stream
    stream.stop_stream()
    stream.close()
    #p.terminate()

    #test audio read
    #wav.write('out.wav',RATE,numpydata)

    # Input waveform : current audio segment stored as a numpy float32 array
    waveform = numpydata
    # Interpreter setup
    interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()
    scores, embeddings, spectrogram = (
        interpreter.get_tensor(scores_output_index),
        interpreter.get_tensor(embeddings_output_index),
        interpreter.get_tensor(spectrogram_output_index))

    #Sort 'scores' array
    prediction = np.mean(scores, axis=0)
    #Choose top 3 values from sorted array
    top3 = np.argsort(prediction)[::-1][:3]
    #print 'class names' and 'confidence value' according top3
    print('\n'.join('  {:12s}: {:.3f}'.format(class_names[i], prediction[i])
                        for i in top3))
    print(' ')
    loopcounter = loopcounter+1

p.terminate()
