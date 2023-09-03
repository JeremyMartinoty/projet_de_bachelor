import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import converter, instrument, note, chord, stream

# Charger le fichier MIDI et prétraiter les données musicales
def preprocess_data(file_path, sequence_length=100):
    notes = []

    midi = converter.parse(file_path)
    notes_to_parse = None

    try:
        # Fichier MIDI avec plusieurs parties
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:
        # Fichier MIDI avec une seule partie
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for str_ in element.normalOrder))

    unique_notes = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(unique_notes))

    input_sequences = []
    output_sequences = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        input_sequences.append([note_to_int[note] for note in sequence_in])
        output_sequences.append(note_to_int[sequence_out])

    num_unique_notes = len(unique_notes)
    num_sequences = len(input_sequences)

    # Convertir les séquences en tableaux numpy
    input_data = np.reshape(input_sequences, (num_sequences, sequence_length, 1))
    input_data = input_data / float(num_unique_notes)

    output_data = keras.utils.to_categorical(output_sequences)

    return input_data, output_data, num_unique_notes, note_to_int

# Définir l'architecture du modèle Transformer
def create_transformer_model(sequence_length, num_unique_notes):
    input_layer = keras.layers.Input(shape=(sequence_length, 1))
    transformer_layer = keras.layers.Transformer(
        num_layers=4,
        d_model=256,
        num_heads=8,
        activation='relu',
        dropout=0.2
    )(input_layer)
    flatten_layer = keras.layers.Flatten()(transformer_layer)
    dense_layer = keras.layers.Dense(256, activation='relu')(flatten_layer)
    output_layer = keras.layers.Dense(num_unique_notes, activation='softmax')(dense_layer)

    model = keras.models.Model(input_layer, output_layer)
    return model
# Générer de la musique avec le modèle entraîné
def generate_music(seed_sequence, model, num_unique_notes, note_to_int, sequence_length=100, num_notes=100):
    generated_sequence = seed_sequence.copy()

    for _ in range(num_notes):
        input_sequence = [note_to_int[note] for note in generated_sequence[-sequence_length:]]
        input_sequence = np.reshape(input_sequence, (1, sequence_length, 1))
        input_sequence = input_sequence / float(num_unique_notes)
        predicted_output = model.predict(input_sequence)
        predicted_note_index = np.argmax(predicted_output)
        predicted_note = list(note_to_int.keys())[list(note_to_int.values()).index(predicted_note_index)]
        generated_sequence.append(predicted_note)

    return generated_sequence

# Convertir la séquence générée en fichier MIDI
def convert_to_midi(generated_sequence, output_file):
    output_stream = stream.Stream()

    for element in generated_sequence:
        notes = element.split('.')
        if len(notes) > 1:  # Accord
            chord_notes = [note.Note(int(note_)) for note_ in notes]
            chord_ = chord.Chord(chord_notes)
            output_stream.append(chord_)
        else:  # Note unique
            note_ = note.Note(int(notes[0]))
            output_stream.append(note_)

    output_stream.write('midi', fp=output_file)

def main():
    # Charger les données et créer le modèle
    file_path = './dataset/classical_music_midi/chopin/'
    sequence_length = 100

    input_data, output_data, num_unique_notes, note_to_int = preprocess_data(file_path, sequence_length)
    model = create_transformer_model(sequence_length, num_unique_notes)

    # Compiler le modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Entraîner le modèle
    model.fit(input_data, output_data, epochs=50, batch_size=128)

    # Générer de la musique avec le modèle entraîné
    seed_sequence = ['C4', 'E4', 'G4']
    generated_sequence = generate_music(seed_sequence, model, num_unique_notes, note_to_int)

    # Convertir la séquence générée en fichier MIDI
    output_file = './output/midi/' + 'transformer.mid'
    convert_to_midi(generated_sequence, output_file)

if __name__ == '__main__':
    main()
