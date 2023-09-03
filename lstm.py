import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from music21 import converter, instrument, note, chord, stream, midi
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

TRAIN = True

def get_all_midi_files(data_path):
    all_midis= []
    for file in os.listdir(data_path):
        if file.endswith(".mid"):
            tr = data_path + file
            print(tr)
            midi = converter.parse(tr)
            all_midis.append(midi)
    return all_midis

def get_notes(all_midis):
    notes = []
    for midi in all_midis:
        notes_to_parse = None
        # S'il y a plusieurs parties, on prend la première
        parts = midi.parts.stream()
        if len(parts) > 1:
            notes_to_parse = parts[0].recurse()
        else:
        # Sinon, on prend toutes les notes
            notes_to_parse = midi.flat.notes
        # On parcourt toutes les notes
        for element in notes_to_parse:
            # Si c'est une note, on l'ajoute à la liste
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            # Si c'est un accord, on l'ajoute à la liste
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

#Prépare les séquences d'entrée/sortie pour le réseau LSTM"""
def prepare_sequences(notes, sequence_length, n_vocab):
    # Tableau des notes
    pitchnames = sorted(set(item for item in notes))
    # Tableau d'entrée
    network_input = []
    # Tableau de sortie
    network_output = []
    # On parcourt toutes les notes
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitchnames.index(char) for char in sequence_in])
        network_output.append(pitchnames.index(sequence_out))
    n_patterns = len(network_input)
    # On redimensionne le tableau d'entrée
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # On normalise les données
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)
    return (network_input, network_output, pitchnames)

#Crée le modèle LSTM"""
def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

#Entraîne le modèle
def train_model(model, network_input, network_output, epochs, batch_size, weights_file):
    checkpoint = ModelCheckpoint(weights_file, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    return model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

def generate_notes(model, network_input, pitchnames, n_vocab, sequence_length, num_notes):
    # On choisit une séquence aléatoire
    start = np.random.randint(0, len(network_input)-1)
    # On récupère la séquence
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = network_input[start]
    prediction_output = []
    # On génère les notes
    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]
    return prediction_output 

# Création de la séquence musicale
def create_midi(prediction_output, directory):
    #Crée un fichier MIDI à partir des notes générées
    offset = 0
    output_notes = []
    # On parcourt toutes les notes générées
    for pattern in prediction_output:
        # Si c'est une note
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                # On récupère la note
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            # On crée un accord
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Si c'est une pause
        else:
            # On crée une note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # On incrémente l'offset
        offset += 0.5
    # On crée un Stream
    midi_stream = stream.Stream(output_notes)
    # On écrit le fichier MIDI
    i=0
    # On vérifie que le fichier n'existe pas déjà
    filename = directory + 'output' + str(i)
    while os.path.exists(filename + '.mid'):
        i+=1
        filename = directory + 'output' + str(i) + '.mid'
    midi_stream.write('midi', fp=filename)
    print("Fichier créé : " + filename)
    return filename

def plot_loss(history):
    # Visualisation de la perte (loss)
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

def plot_accuracy(history):
    # Visualisation de la précision (accuracy)
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

def show_midi_file(filepath):
    from music21 import midi
    mf = midi.MidiFile()
    mf.open(filepath)
    mf.read()
    mf.close()

def main():
    global TRAIN
    # Paramètres
    input_midi_dir = './dataset/classical_music_midi/chopin/'
    weights_file = './output/weights/lstm_weights.hdf5'
    weights_file_save = './output/weights_save/lstm_weights.hdf5'
    output_midi_dir = './output/midi/'

    sequence_length = 100 # Nombre de notes à prendre en compte pour prédire la suivante
    num_notes = 100 # Nombre de notes à générer
    epochs = 5 # Nombre d'itérations
    batch_size = 64 
    #Si il y a un argument "train" on entraîne le modèle
    if len(sys.argv) > 1 : 
        if sys.argv[1] == "train":
            TRAIN = True
    
    print('Chargement des fichiers MIDI...')
    all_midis = get_all_midi_files(input_midi_dir)
    print('Nombre de fichiers MIDI chargés: ', len(all_midis))

    # Transforme les fichiers MIDI en notes numériques
    print('Chargement des notes...')
    notes = get_notes(all_midis)
    print('Nombre de notes chargées: ', len(notes))
    print("10 premières notes :", notes[:10])
    # Nombre de notes uniques
    n_vocab = len(set(notes))

    # Préparation des séquences d'entrée/sortie
    print('Préparation des séquences d\'entrée/sortie...')
    network_input, network_output, pitchnames = prepare_sequences(notes, sequence_length, n_vocab)

    # Création du modèle
    print('Création du modèle...')
    model = create_model(network_input, n_vocab)
    # Entraînement du modèle
    if TRAIN:
        print('Entraînement du modèle...')
        history = train_model(model, network_input, network_output, epochs, batch_size, weights_file)
        plot_loss(history)
        plot_accuracy(history)
    # Chargement des poids
    model.load_weights(weights_file_save)

    # Génération des notes
    print('Génération des notes...')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, sequence_length, num_notes)
    print('Nombre de notes générées: ', len(prediction_output))
    # Création du fichier MIDI
    filename = create_midi(prediction_output, output_midi_dir)
    # Affichage du fichier MIDI
    show_midi_file(filename)

if __name__ == '__main__':
    main()
