import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np


class AmpEnvExtractor():
    """
    This class extracts the amplitude envelope of a signal

    The amplitude envelope refers to the changes in amplitude
    of a signal over time.

    Perceptually, the amplitude envelope is the perceived loudness.
    """

    def __init__(self, folder_path, genres):
        self.folder_path = folder_path
        self.genres = genres
        self.feature_name = 'amp_enve'

    def load_all_audio_for_all_genres(self):
        """
        Description: This function loads all audio files for all genres

        Parameters:
                    None

        Returns:
                    genres_dict (dict): A dictionary of audio files for all genres
        """
        genres_dict = {}
        for genre in self.genres:
            audio_dict = self.load_all_audio_for_genre(genre)
            genres_dict[genre] = audio_dict

        return genres_dict

    def load_all_audio_for_genre(self, genre):
        """
        Description: This function loads all audio files in a folder

        Returns:
                    audio_dict (dict): A dictionary of audio files
        """
        audio_dict = {}

        for audio_file in os.listdir(os.path.join(self.folder_path, genre)):
            audio, sr = librosa.load(os.path.join(
                self.folder_path, genre, audio_file))
            audio_dict[audio_file] = audio
        return audio_dict

    def load_audio(self, audio_path):
        """
        Description: This function loads the audio file

        Parameters:
                    audio_path (str): The path to the audio file
        Returns:
                    audio (np.array): The audio file as a numpy array
                    sr (int): The sampling rate of the audio file
        """

        # Load the audio file as a floating point time series
        # Target sampling rate is 44.1 kHz
        # audio_path = self.folder_path + audio_filename
        audio, sr = librosa.load(audio_path, sr=44100)

        # audio: np.ndarray [shape=(n,) or (2, n)]
        # sr: number > 0 [scalar]
        return audio, sr

    def get_duration(self, audio, sr):
        """This function returns the duration of the audio file"""

        # Get the duration of the audio file
        duration = librosa.get_duration(y=audio, sr=sr)
        return duration

    def visualize_audio(self, audio):
        """This function visualizes the audio file"""

        # Visualize the audio file
        plt.figure(figsize=(15, 17))
        librosa.display.waveshow(audio)
        plt.show()

    def visualize_multi_audio(self, audio_song_names, audio_bundle):
        """
        Description: This function overlays mutliple audio files on top of each other

        Parameters:
                    audio_bundle (list): A list of audio files
        Returns:
                    None
        """

        plt.figure(figsize=(15, 17))

        row = 1
        for audio in audio_bundle:
            plt.subplot(3, 1, row)
            librosa.display.waveshow(audio)
            plt.ylim((-1, 1))
            plt.title(audio_song_names[row-1])
            row += 1

        plt.show()

    def get_amp_enve(self, audio, frame_size, hop_length):
        """Compute the amplitude envelope of an audio signal

        Args:
            audio (ndarray): The audio signal
            frame_size (int): The size of the frame
            hop_length (int): The number of samples between successive frames
        """
        ae = []

        # Calculate AE for each frame
        for i in range(0, len(audio), hop_length):
            curr_frame = max(audio[i:i+frame_size])
            ae.append(curr_frame)

        return np.array(ae)

    def get_amp_enve_advanced(self, audio, frame_size, hop_length):
        """Compute the amplitude envelope of an audio signal

        Args:
            audio (ndarray): The audio signal
            frame_size (int): The size of the frame
            hop_length (int): The number of samples between successive frames
        """
        return np.array(([max(audio[i:i+frame_size]) for i in range(0, len(audio), hop_length)]))


def main():

    # Print current working directory
    print("Current working directory: ", os.getcwd())

    # Path to data containing genre folders
    raw_data_path = os.path.join(os.getcwd(), 'data', 'raw')

    print("Raw data path: ", raw_data_path)
    # Path to data containing genre folders
    genres = ["classical", "country", "disco", "hiphop",
              "jazz", "metal", "pop", "reggae", "rock"]

    # Save training data to interim folder
    output_path = os.path.join(os.getcwd(), 'data', 'interim')

    # Create an instance of the AmpEnvExtractor class
    Extractor = AmpEnvExtractor(raw_data_path, genres)

    # Process Audio Files for one genre
    classical_dict = Extractor.load_all_audio_for_genre('classical')
    rock_dict = Extractor.load_all_audio_for_genre('rock')

    # Print the number of audio files in the classical genre
    print("Number of audio files in the classical genre: ", len(classical_dict))

    # Print the first audio file in the classical genre
    print("First audio file in the classical genre: ",
          list(classical_dict.keys())[0])

    # Print the information of the first audio file in the classical genre
    print("Information of the first audio file in the classical genre: ",
          classical_dict[list(classical_dict.keys())[0]])

    # Print the duration of 1 sample of the first audio file in the classical genre
    sample_duration = Extractor.get_duration(
        classical_dict[list(classical_dict.keys())[0]], 44100)
    print("Duration of first audio file in the classical genre: ", sample_duration)

    # Visualize the first audio file in the classical genre
    # Extractor.visualize_audio(classical_dict[list(classical_dict.keys())[0]])

    # Overlay the first audio file in the classical genre and the first audio file in the rock genre
    audio_bundle = [classical_dict[list(classical_dict.keys())[
        0]], rock_dict[list(rock_dict.keys())[0]]]
    audio_song_names = ['classical', 'rock']
    Extractor.visualize_multi_audio(
        audio_song_names=audio_song_names, audio_bundle=audio_bundle)

    # Get the amplitude envelope of the first audio file in the classical genre
    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    ae_array_classical = Extractor.get_amp_enve_advanced(classical_dict[list(
        classical_dict.keys())[0]], FRAME_SIZE, HOP_LENGTH)
    print("Amplitude envelope of the first audio file in the classical genre: ", len(
        ae_array_classical))

    # Get the amplitude envelope of the first audio file in the rock genre
    ae_array_rock = Extractor.get_amp_enve_advanced(rock_dict[list(
        rock_dict.keys())[0]], FRAME_SIZE, HOP_LENGTH)
    print("Amplitude envelope of the first audio file in the rock genre: ",
          len(ae_array_rock))

    # Vizualize the amplitude envelope of the first audio file in the classical genre
    ae_audio_bundle = [ae_array_classical, ae_array_rock]
    ae_audio_song_names = ['ae_array_classical', 'ae_array_rock']
    Extractor.visualize_multi_audio(ae_audio_song_names, ae_audio_bundle)


if __name__ == "__main__":
    main()
