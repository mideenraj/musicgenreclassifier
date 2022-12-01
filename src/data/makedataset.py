import os
import numpy as np
from src.data.PrepareInput import PrepareAudio
import json
from collections import deque
from tqdm import tqdm


class MusicTrainingData:
    """
    The MusicTrainingData class generates a training dataset
    to be used on the neural network. This will output a NumPy
    array file (.npy) that contains a mel spectrograph and a
    label for the spectrograph as a one-hot vector to signal
    the correct genre.
    """

    def __init__(self):
        """
        Constructor method to create blank training data list
        and genre dictionary.
        """
        self.training_data = deque()
        self.genre_dict = {}

    def create_genre_dictionary(self, data_path, output_path):
        """
        Creates a dictionary using the names of the objects at a
        specified path. The objects should be directories that
        contain data for a single music genre (ex. Rap, Classical).
        """
        self.genre_dict = {}
        genre_count = 0
        for g in os.listdir(data_path):
            self.genre_dict[g] = genre_count
            genre_count += 1
        # Save dictionary to output path
        self.save_genre_dict(output_path)

    def get_genre_dictionary(self):
        """
        Returns genre dictionary
        """
        return self.genre_dict

    def make_training_data(self, data_path, output_path):
        """
        Creates numpy array of mel spectrograph and genre label using
        the genre dictionary to create a one-hot vector. Processes
        all files within genre-labeled directories at a specified path.
        """
        self.create_genre_dictionary(data_path, output_path)
        genre_count = len(self.genre_dict)
        mel_spectrogram = PrepareAudio()
        # Iterate through all genres
        for genre in self.genre_dict:
            # For each file in a genre
            for f in (pbar := tqdm(os.listdir(os.path.join(data_path, genre)))):
                # Progress bar description
                pbar.set_description(f"Processing {genre}")
                # Use Librosa to create a spectrograph - Midhun's code
                img = mel_spectrogram.start(os.path.join(data_path, genre, f))
                # Add image and label to training data
                label = np.eye(genre_count)[self.genre_dict[genre]]
                self.training_data.append([img, list(label)])

        # Shuffle and save dataset to designated output path
        self.training_data = np.array(self.training_data)
        np.random.shuffle(self.training_data)
        np.save(os.path.join(output_path, 'training_data.npy'), self.training_data)

    def save_genre_dict(self, output_path):
        """
        Saves the genre dictionary to a specified output path.
        """
        file_path = os.path.join(output_path, 'genre_dict.txt')
        with open(file_path, 'w') as dict_file:
            dict_file.write(json.dumps(self.genre_dict))
