import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from music21 import converter, instrument, note, chord, stream, midi
from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
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

# Paramètres du GAN

def generate_latent_points(num_samples, latent_dim):
    """
    Génère des points aléatoires dans l'espace latent comme entrée pour le générateur.
    """
    return np.random.randn(num_samples, latent_dim)

def build_generator(dataset):
    num_units = 256
    num_time_steps = 32
    generator = Sequential()
    print("dataset.shape[1] : ", dataset.shape[1])
    generator.add(LSTM(num_units, input_shape=(num_time_steps, dataset.shape[1])))
    generator.add(RepeatVector(num_time_steps))
    generator.add(LSTM(num_units, return_sequences=True))
    generator.add(TimeDistributed(Dense(dataset.shape[1], activation='sigmoid')))
    generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    return generator

def build_discriminator(shape):
    """
    Construit le discriminateur.
    """
    model = Sequential()
    model.add(Dense(1024, input_dim=shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def build_gan(generator, discriminator, dataset):
    """
    Construit le modèle GAN combinant le générateur et le discriminateur.
    """
    discriminator.trainable = False
    discriminator.build(generator.output_shape)
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    
    return model

def train_gan(generator, discriminator, gan, dataset, num_epochs, batch_size, latent_dim, min_value, max_value):
    """
    Entraîne le GAN sur le jeu de données.
    """
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    loss_history = []
    
    for epoch in range(num_epochs):
        # Entraînement du discriminateur
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_chords = dataset[idx]
        noise = generate_latent_points(batch_size, latent_dim)
        fake_chords = generator.predict(noise)
        
        discriminator_loss_real = discriminator.train_on_batch(real_chords, real)
        discriminator_loss_fake = discriminator.train_on_batch(fake_chords, fake)
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
        # Entraînement du générateur
        num_batches = len(dataset) // batch_size
        for batch in range(num_batches):
            X_real = dataset[batch * batch_size : (batch + 1) * batch_size]
            X_fake = generator.predict(np.random.randn(batch_size, latent_dim))
            generator_loss = generator.train_on_batch(X_fake, X_real)

        print(f'Epoch: {epoch}, Generator Loss: {generator_loss}')
        print(f'Epoch: {epoch}, Discriminator Loss: {discriminator_loss}')
        loss_history.append([generator_loss, discriminator_loss])
    print('Entraînement terminé.')
    generator.save('generator.h5')
    print('Modèle sauvegardé.')
    #return loss history and accuracy history
    return loss_history

def generate_music_gan(generator, latent_dim):
    """
    Génère de la musique en utilisant le générateur GAN entraîné.
    """
    num_samples = int(input("Nombre de morceaux à générer : "))

    for i in range(num_samples):
        num_chords = 16  # Nombre d'accords dans la séquence
        chords = []

        for _ in range(num_chords):
            noise = generate_latent_points(1, latent_dim)
            generated_chords = generator.predict(noise)
            print("generated_chords : ", generated_chords)
            chords.append(generated_chords)

        chords = np.concatenate(chords, axis=0)
        print("chords : ", chords)
        midi_filepath = f'generated_music_gan_{i}.mid'
        generate_midi_file(chords, midi_filepath)
        print(f"Musique générée : {midi_filepath}")

def generate_midi_file(chords, filepath):
    """
    Génère un fichier MIDI à partir d'une séquence d'accords.
    """
    # On transforme les accords en notes
    notes = utils.chords_to_notes(chords)
    # On crée un objet stream de music21
    midi_stream = stream.Stream()
    # On parcourt toutes les notes
    for note in notes:
        # Si c'est une note
        if ('.' in note) or note.isdigit():
            # On crée un objet Note
            notes_in_chord = note.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            midi_stream.append(new_chord)
        # Si c'est un accord
        else:
            new_note = note.Note(note)
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)
    # On écrit le stream dans un fichier MIDI
    midi_stream.write('midi', fp=filepath)


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
    weights_file_save = 'generator.h5'
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
    generator = build_generator(network_input)
    discriminator = build_discriminator(generator.output_shape)
    model = build_gan(generator, discriminator, network_input)
    # Entraînement du modèle
    if TRAIN:
        print('Entraînement du modèle...')
        generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        history = train_gan(generator, discriminator, model, network_input, epochs, batch_size, 100, 0, 1)
        plot_loss(history)
        plot_accuracy(history)
    # Chargement des poids
    model.load_weights(weights_file_save)
    generate_music_gan(generator, 100)

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
