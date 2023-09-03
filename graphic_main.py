import os
import gan
import midi_utils
import numpy as np
from tensorflow.keras.models import load_model


# Chemin vers le répertoire contenant les fichiers MIDI d'apprentissage
midi_dir = './input/classical_music_midi/chopin/'

def print_main_menu():
    print("\nMenu principal:")
    print("1. Entraîner un modèle")
    print("2. Générer de la musique")
    print("3. Écouter des morceaux déjà générés")
    print("0. Quitter")

def print_train_menu():
    print("\nMenu d'entraînement :")
    print("1. GAN")
    print("2. Transformer")
    print("0. Retour")

def print_generate_menu():
    print("\nMenu de génération :")
    print("1. GAN")
    print("2. Transformer")
    print("0. Retour")

def print_listen_menu():
    print("\nMenu d'écoute :")
    print("1. Écouter des morceaux générés par le GAN")
    print("2. Écouter des morceaux générés par le Transformer")
    print("0. Retour")

def main():
    chords_data = midi_utils.load_midi_files(midi_dir)
    dataset = midi_utils.normalize_data(chords_data)
    print("Nombre de notes chargées :", len(dataset))
    print("Les 10 premières notes :", dataset[:10])
    
    while True:
        print_main_menu()
        choice = input("Choisissez une option (0-3) : ")
        
        if choice == "1":
            while True:
                print_train_menu()
                train_choice = input("Choisissez un modèle à entraîner (0-2) : ")
                
                if train_choice == "1":
                    latent_dim = int(input("Dimension de l'espace latent : "))
                    num_epochs = int(input("Nombre d'époques d'entraînement : "))
                    batch_size = int(input("Taille du lot (batch size) : "))
                    
                    generator = gan.build_generator(dataset)
                    discriminator = gan.build_discriminator()
                    gan_model = gan.build_gan(generator, discriminator)
                    generator.compile(loss='binary_crossentropy', optimizer=gan.Adam(learning_rate=0.0002, beta_1=0.5))
                    discriminator.compile(loss='binary_crossentropy', optimizer=gan.Adam(learning_rate=0.0002, beta_1=0.5))
                    gan_model.compile(loss='binary_crossentropy', optimizer=gan.Adam(learning_rate=0.0002, beta_1=0.5))
                    gan.train_gan(generator, discriminator, gan_model, dataset, num_epochs, batch_size, latent_dim)
                    
                    break
                
                elif train_choice == "2":
                    # Ajoutez ici le code pour l'entraînement du Transformer
                    # transformer.train_transformer()  # Exemple de fonction d'entraînement du Transformer
                    
                    break
                
                elif train_choice == "0":
                    break
                
                else:
                    print("Option invalide. Veuillez choisir une option valide.")
        
        elif choice == "2":
            while True:
                print_generate_menu()
                generate_choice = input("Choisissez un modèle pour générer de la musique (0-2) : ")
                
                if generate_choice == "1":
                    # Ajoutez ici le code pour la génération de musique avec le GAN
                    generator = load_model('generator.h5')
                    gan.generate_music_gan(generator,32)  # Exemple de fonction de génération de musique avec le GAN
                    
                    break
                
                elif generate_choice == "2":
                    # Ajoutez ici le code pour la génération de musique avec le Transformer
                    # transformer.generate_music_transformer()  # Exemple de fonction de génération de musique avec le Transformer
                    
                    break
                
                elif generate_choice == "0":
                    break
                
                else:
                    print("Option invalide. Veuillez choisir une option valide.")
        
        elif choice == "3":
            while True:
                print_listen_menu()
                listen_choice = input("Choisissez une option d'écoute (0-2) : ")
                
                if listen_choice == "1":
                    # Ajoutez ici le code pour écouter des morceaux générés par le GAN
                    # gan.listen_music_gan()  # Exemple de fonction pour écouter des morceaux générés par le GAN
                    
                    break
                
                elif listen_choice == "2":
                    # Ajoutez ici le code pour écouter des morceaux générés par le Transformer
                    # transformer.listen_music_transformer()  # Exemple de fonction pour écouter des morceaux générés par le Transformer
                    
                    break
                
                elif listen_choice == "0":
                    break
                
                else:
                    print("Option invalide. Veuillez choisir une option valide.")
        
        elif choice == "0":
            print("Au revoir !")
            break
        
        else:
            print("Option invalide. Veuillez choisir une option valide.")

if __name__ == "__main__":
    main()
