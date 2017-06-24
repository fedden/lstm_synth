from __future__ import absolute_import
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import dropout
import numpy as np
import phase_gen
import librosa
import tflearn
import random
import os

data_path = u"/home/lstm_synth/assets/flute"
checkpoint = u"-4348-99000"
fft_size = 512
window_size = 256
hop_size = 128
sequence_length = 10 
np.random.seed(8)
training = True
learning_rate = 0.001
training_iters = 500
batch_size = 64
number_hidden = 128
highway_size = 64
highway_layer_amount = 15
sequence_max_length = 2000
amount_generated_sequences = 20
tf_id = u'lstm_' + unicode(number_hidden) + u'_highway_size_' + unicode(highway_size) \
        + u'_layer_' + unicode(highway_layer_amount) + u'_lr_' + unicode(learning_rate)


def unison_shuffled_copies(a, b):
    u""" Shuffle NumPy arrays in unison. """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


checkpoint_path = u'models/' + tf_id + u"/checkpoints"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

audio_path = u'models/' + tf_id + u'/audio'
if not os.path.exists(audio_path):
    os.makedirs(audio_path)

x_frames = []
y_frames = []

if os.path.exists(data_path):

    file_names = os.listdir(data_path)
    print u"found", len(file_names), u"files"

    fft_frames = []

    for file in file_names:
        if file.endswith(u'.wav'):

            file = os.path.join(data_path, file)
            data, sample_rate = librosa.load(file, sr=44100, mono=True)
            mags_phases = librosa.stft(data, n_fft=fft_size,
                                       win_length=window_size,
                                       hop_length=hop_size)
            magnitudes, phases = librosa.magphase(mags_phases)

            for magnitude_bins in magnitudes.T:
                fft_frames += [magnitude_bins]

    for s in xrange(sequence_length):
        start = s
        end = len(fft_frames) - sequence_length - 1 - s
        step = sequence_length + 1
        for i in xrange(start, end, step):
            x_frames += [fft_frames[i:i + sequence_length]]
            y_frames += [fft_frames[i + sequence_length + 1]]

x_frames = np.array(x_frames)
y_frames = np.array(y_frames)

x_frames, y_frames = unison_shuffled_copies(x_frames, y_frames)

split = int(0.05 * x_frames.shape[0])
valid_x = x_frames[0:split]
valid_y = y_frames[0:split]
train_x = x_frames[split:]
train_y = y_frames[split:]

print x_frames.shape
print x_frames.shape[1], x_frames.shape[2]
net = tflearn.input_data([None, x_frames.shape[1], x_frames.shape[2]])
print net.get_shape().as_list()
net = bidirectional_rnn(net, BasicLSTMCell(number_hidden),
                        BasicLSTMCell(number_hidden))
net = dropout(net, 0.8)
fc = tflearn.fully_connected(net, highway_size, activation='elu',
                             regularizer='L2', weight_decay=0.001)
net = fc

for i in xrange(highway_layer_amount):
    net = tflearn.highway(net, highway_size, activation='elu',
                          regularizer='L2', weight_decay=0.001,
                          transform_dropout=0.8)

net = tflearn.fully_connected(net, y_frames.shape[1], activation='elu')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='mean_square')


class MonitorCallback(tflearn.callbacks.Callback):

    def __init__(self, model, tf_id):
        self.lowest_loss = np.inf
        self.number_saves = 0
        self.model = model
        self.tf_id = tf_id

    def on_batch_end(self, training_state, snapshot=False):
    	if training_state.global_loss < self.lowest_loss:
            self.lowest_loss = training_state.global_loss
            self.number_saves += 1
            self.model.save(self.tf_id + '_' + str(self.number_saves)  + '.tfl')
            

cp = os.path.join(checkpoint_path, (tf_id + u'.ckpt' + checkpoint))
model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path=cp,
                    max_checkpoints=0)
callback = MonitorCallback(model, os.path.join(checkpoint_path, tf_id))
if training:
    model.fit(train_x, train_y, validation_set=((valid_x, valid_y)),
              show_metric=True, batch_size=batch_size, n_epoch=training_iters,
              snapshot_epoch=False, snapshot_step=1000, run_id=tf_id,
              callbacks=callback)
else:
    model.load(cp)

for i in xrange(amount_generated_sequences):
    random_index = random.randint(0, (len(train_x) - 1))

    impulse = np.array(train_x[random_index])
    predicted_magnitudes = impulse
    for j in xrange(sequence_max_length):
        impulse = np.array(impulse).reshape(1, x_frames.shape[1],
                                            x_frames.shape[2])
        prediction = model.predict(impulse)
        predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))
        impulse = predicted_magnitudes[-sequence_length:]

    predicted_magnitudes = np.array(predicted_magnitudes)
    print i, predicted_magnitudes.shape
    phases = phase_gen.gen_phases(predicted_magnitudes.shape[0], fft_size,
                                  hop_size, sample_rate)
    audio = phase_gen.fft2samples(predicted_magnitudes, phases, hop_size)
    maxv = np.iinfo(np.int16).max
    audio_wav = (librosa.util.normalize(audio) * maxv).astype(np.int16)
    audio_name = tf_id + u'_' + unicode(i) + u'.wav'
    librosa.output.write_wav(audio_path + u'/' + audio_name, audio_wav,
                             sample_rate, norm=False)
